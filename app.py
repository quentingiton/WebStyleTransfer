import streamlit as st
import modal
from PIL import Image
import datetime

def get_time_until_next_month():
    now = datetime.datetime.now()
    if now.month == 12:
        next_month = datetime.datetime(now.year + 1, 1, 1)
    else:
        next_month = datetime.datetime(now.year, now.month + 1, 1)
    delta = next_month - now
    return f"{delta.days} jours et {delta.seconds // 3600} heures"

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
        weight = st.sidebar.slider("Importance du contenu de l'image source", 0.001, 4, 1.0)

        img = Image.open(source_file)
        long_edge = max(img.width, img.height)
        default_resize = max(256, long_edge)

        resize_to = st.sidebar.number_input("Réduire l'image à (min 256) pixels", min_value=256, value=default_resize)

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
         st.info("Temps estimé : moins de 30 secondes.")
         iters = st.sidebar.slider("Nombre d'itérations souhaitées", 10, 100, 40)
         step = st.sidebar.slider("Taille du pas (par itération) souhaité", 0.1, 2.0, 1.0)
         apply_reg = st.sidebar.checkbox("Appliquer la régularisation", value=False)

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