import os
import json
import modal
import datetime
import numpy as np
from PIL import Image
import streamlit as st

COST_SEC_STYLE = (0.59 / 3600) + (2 * 0.0473 / 3600) + (4 * 0.0080 / 3600)
COST_SEC_COLOR = (0.0473 / 3600) + (0.0080 / 3600)

BALANCE_FILE = "pseudo_balance.json"
INITIAL_BALANCE = 29.72

def get_balance():
    if not os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "w") as f:
            json.dump({"balance": INITIAL_BALANCE}, f)
        return INITIAL_BALANCE
    
    with open(BALANCE_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get("balance", INITIAL_BALANCE)
        except json.JSONDecodeError:
            return INITIAL_BALANCE

def deduct_balance(amount):
    current = get_balance()
    new_balance = max(0.0, current - amount)
    with open(BALANCE_FILE, "w") as f:
        json.dump({"balance": new_balance}, f)
    return new_balance

def format_time(total_seconds):
    total_seconds = int(round(total_seconds))
    days = total_seconds // 86400
    total_seconds %= 86400
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days} jour{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} heure{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} min")
    if seconds > 0 or not parts:
        parts.append(f"{seconds} s")

    return " ".join(parts)

def get_time_until_next_month():
    now = datetime.datetime.now()
    if now.month == 12:
        next_month = datetime.datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime.datetime(now.year, now.month + 1, 1)
    delta = next_month - now
    return format_time(delta.total_seconds())

def estimate_style_time(original_width, original_height, resize_to):
    # Calculate Aspect Ratio
    long_edge = max(original_width, original_height)
    short_edge = min(original_width, original_height)
    A = long_edge / short_edge if short_edge > 0 else 1.0
    
    P = (resize_to ** 2) / A
    
    k_base = 8.0     # Container/CUDA init overhead
    K_conv = 1.8e-4  # Forward/Backward pass time per pixel across all scales
    K_loss = 35.0    # Fixed cost of 200 iterations * S scales of 1024x5000 distmat

    est_time = round(k_base + (K_conv * P) + K_loss)
    est_cost = est_time * COST_SEC_STYLE

    return est_time, est_cost

def estimate_colour_time(source_width, source_height, target_width, target_height, iters, apply_reg):
    N = source_width * source_height
    M = target_width * target_height
    
    c_base = 0.5
    c_sort = 1.2e-7 # seconds per N log N operation
    c_reg = 4.5e-7  # seconds per pixel for uniform_filter
    
    term_N = N * np.log2(N)
    term_M = M * np.log2(M)
    
    time_iters = iters * 3 * c_sort * (term_N + term_M)
    time_reg = (c_reg * N) if apply_reg else 0.0

    est_time = round(c_base + time_iters + time_reg, 1)
    est_cost = est_time * COST_SEC_COLOR

    return est_time, est_cost

st.title("Transfert d'images")

st.sidebar.header("Portefeuille Modal")
balance_placeholder = st.sidebar.empty()
current_balance = get_balance()
balance_placeholder.metric(label="Solde restant estimé", value=f"${current_balance:.4f}")
st.sidebar.markdown("---")

app_mode = st.sidebar.selectbox("Choix du mode", ["Transfert de style", "Transfert de couleur"])

col1, col2 = st.columns(2)
with col1:
    source_file = st.file_uploader("Choisir une image à modifier", type=["jpg", "png", "jpeg"])
with col2:
    target_file = st.file_uploader("Choisir une image de style/couleur", type=["jpg", "png", "jpeg"])

if source_file and target_file:
    st.image([source_file, target_file], width=300, caption=["Source", "Cible"])

    source_bytes = source_file.getvalue()
    target_bytes = target_file.getvalue()

    if app_mode == "Transfert de style":
        weight = st.sidebar.slider("Importance du contenu de l'image source", 0.001, 8.0, 1.0)

        img = Image.open(source_file)
        long_edge = max(img.width, img.height)
        default_resize = max(256, long_edge)

        resize_to = st.sidebar.number_input("Réduire l'image à (min 256) pixels", min_value=256, value=default_resize)

        estimated_time, estimated_cost = estimate_style_time(img.width, img.height, resize_to)
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Temps estimé (T4 GPU)", value=f"~{format_time(estimated_time)}")
        m_col2.metric(label="Coût estimé", value=f"~${estimated_cost:.2f}")

        if st.button("Lancer le transfert de style"):
            with st.spinner("Transfert en cours..."):
                try:
                    f = modal.Function.from_name("image-transfer", "run_style_transfer")
                    result_bytes = f.remote(source_bytes, target_bytes, weight, resize_to)
                    st.image(result_bytes, caption="Résultat")

                    st.download_button(
                        label = "Télécharger le résultat",
                        data = result_bytes,
                        file_name = "resultat_transfert_style.png",
                        mime = "image/png"
                    )

                    new_balance = deduct_balance(estimated_cost)
                    balance_placeholder.metric(label="Solde restant estimé", value=f"${new_balance:.4f}")


                except Exception as e:
                    error_msg = str(e).lower()
                    if "credit" in error_msg  or "exhausted" in error_msg or "limit" in error_msg:
                        st.error(
                            f"Quota mensuel Modal dépassé. Les transferts seront de nouveau "
                            f"disponibles dans **{get_time_until_next_month()}**"
                        )
                    else:
                        st.error(f"Une erreur inattendue est survenue : {e}")

    elif app_mode == "Transfert de couleur":
        iters = st.sidebar.slider("Nombre d'itérations souhaitées", 10, 100, 40)
        step = st.sidebar.slider("Taille du pas (par itération) souhaité", 0.1, 2.0, 1.0)
        apply_reg = st.sidebar.checkbox("Appliquer la régularisation", value=False)

        source_img = Image.open(source_file)
        target_image = Image.open(target_file)

        estimated_time, estimated_cost = estimate_colour_time(
            source_img.width, source_img.height, 
            target_image.width, target_image.height, 
            iters, apply_reg
            )
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label="Temps estimé (CPU)", value=f"~{format_time(estimated_time)}")
        m_col2.metric(label="Coût estimé", value=f"~${estimated_cost:.5f}")

        if st.button("Lancer le transfert de couleur"):
            with st.spinner("Transfert en cours..."):
                try:
                    f = modal.Function.from_name("image-transfer", "run_color_transfer")
                    result_bytes = f.remote(source_bytes, target_bytes, iters, step, apply_reg)
                    st.image(result_bytes, caption="Résultat")

                    st.download_button(
                        label = "Télécharger le résultat",
                        data = result_bytes,
                        file_name = "resultat_transfert_couleur.png",
                        mime = "image/png"
                    )

                    new_balance = deduct_balance(estimated_cost)
                    balance_placeholder.metric(label="Solde restant estimé", value=f"${new_balance:.4f}")

                except Exception as e:
                    error_msg = str(e).lower()
                    if "credit" in error_msg  or "exhausted" in error_msg or "limit" in error_msg:
                        st.error(
                            f"Quota mensuel Modal dépassé. Les transferts seront de nouveau "
                            f"disponibles dans **{get_time_until_next_month()}**"
                        )
                    else:
                        st.error(f"Une erreur inattendue est survenue : {e}")