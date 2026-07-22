import streamlit as st
import modal
import io

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
        weight = st.sidebar.slider("Importance du contenu de l'image source", 0.1, 2, 1.0)
        resize_to = st.sidebar.number_input("Réduire l'image à (min 256) pixels", min_value=256, value=512)

        if st.button("Lancer le transfert de style"):
            with st.spinner("Transfert en cours..."):
                f = modal.Function.lookup("image-transfer", "run_style_transfer")
                result_bytes = f.remote(source_bytes, target_bytes, weight, resize_to)
                st.image(result_bytes, caption="Résultat")

                st.download_button(
                    label = "Télécharger le résultat",
                    data = result_bytes,
                    file_name = "resultat_transfert_style.png",
                    mime = "image/png"
                )

    elif app_mode == "Transfert de couleur":
         iters = st.sidebar.slider("Nombre d'itérations souhaitées", 10, 100, 40)
         step = st.sidebar.slider("Taille du pas (par itération) souhaité", 0.1, 2.0, 1.0)

         if st.button("Lancer le transfert de couleur"):
            with st.spinner("Transfert en cours..."):
                f = modal.Function.lookup("image-transfer", "run_color_transfer")
                result_bytes = f.remote(source_bytes, target_bytes, iters, step)
                st.image(result_bytes, caption="Résultat")

                st.download_button(
                    label = "Télécharger le résultat",
                    data = result_bytes,
                    file_name = "resultat_transfert_couleur.png",
                    mime = "image/png"
                )