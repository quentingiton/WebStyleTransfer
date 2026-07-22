import streamlit as st
import modal
from PIL import Image
import datetime
import numpy as np

def get_time_until_next_month():
    now = datetime.datetime.now()
    if now.month == 12:
        next_month = datetime.datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime.datetime(now.year, now.month + 1, 1)
    delta = next_month - now
    return f"{delta.days} jours et {delta.seconds // 3600} heures"

def estimate_style_time(original_width, original_height, resize_to):
    # Calculate Aspect Ratio
    long_edge = max(original_width, original_height)
    short_edge = min(original_width, original_height)
    A = long_edge / short_edge if short_edge > 0 else 1.0
    
    P = (resize_to ** 2) / A
    
    k_base = 8.0     # Container/CUDA init overhead
    K_conv = 1.8e-4  # Forward/Backward pass time per pixel across all scales
    K_loss = 35.0    # Fixed cost of 200 iterations * S scales of 1024x5000 distmat
    
    return round(k_base + (K_conv * P) + K_loss)

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
    
    return round(c_base + time_iters + time_reg, 1)

st.title("Transfert d'images")

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

        estimated_time = estimate_style_time(img.width, img.height, resize_to)
        st.info(f"**Temps estimé sur GPU (T4) :** ~{estimated_time} secondes")

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

        estimated_time = estimate_colour_time(source_img.width, source_img.height, target_image.width, target_image.height, iters, apply_reg)
        st.info(f"**Temps estimé sur CPU :** ~{estimated_time} secondes")

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

                except Exception as e:
                    error_msg = str(e).lower()
                    if "credit" in error_msg  or "exhausted" in error_msg or "limit" in error_msg:
                        st.error(
                            f"Quota mensuel Modal dépassé. Les transferts seront de nouveau "
                            f"disponibles dans **{get_time_until_next_month()}**"
                        )
                    else:
                        st.error(f"Une erreur inattendue est survenue : {e}")