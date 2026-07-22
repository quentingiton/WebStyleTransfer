# Image Style and Color Transfer Web App

This repository contains a [web application](https://webstyletransfer.streamlit.app/) for applying advanced style and color transfer to images. The frontend is built with [Streamlit](https://streamlit.io/), and the heavy computational tasks are offloaded to [Modal](https://modal.com/) serverless containers.

## Features

* **Style Transfer (STROTSS):** Uses a deep learning approach to transfer the artistic style of one image to another. This process runs on a Modal T4 GPU for acceleration.
* **Color Transfer:** Applies the color distribution of a target image to a source image using Sliced Optimal Transport. This runs efficiently on Modal CPU containers.
* **Cost & Time Estimation:** Dynamically calculates estimated execution times and serverless costs based on image resolution and algorithmic complexity.
* **Pseudo-Balance Tracking:** Tracks a simulated credit balance locally to monitor estimated cloud spending.

---

## File Structure

* `app.py`: The Streamlit frontend providing the user interface, file uploading, and local cost estimation.
* `modal_backend.py`: The Modal configuration and remote function definitions for executing the transfers in the cloud.
* `strotss.py`: The PyTorch implementation of the Style Transfer by Relaxed Optimal Transport and Self-Similarity algorithm.
* `colour_transfer.py`: The NumPy/SciPy implementation of the Sliced Optimal Transport color transfer, including guided filter regularization.
* `requirements.txt`: Specifies the necessary Python packages (`streamlit` and `modal`).

---

## Live Application (For Users)

If you simply want to transfer the style or color of an image, you do not need to install anything or create a Modal account. 

1. Go to the live web app: [https://webstyletransfer.streamlit.app/](https://webstyletransfer.streamlit.app/)
2. Upload your source image and your style/color target image.
3. Select your desired mode (Style or Color transfer).
4. Adjust the parameters (resolution, iterations, weights) based on the estimated time and cost displayed.
5. Click launch and download your result!

---

## Developer Setup (For Replicating)

If you want to deploy this architecture on your own Modal and Streamlit environments, follow these steps:

### Prerequisites
* A [Modal account](https://modal.com)
* A [Streamlit account](https://streamlit.io)
* Python 3.8+ installed locally

### Installation & Deployment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/quentingiton/WebStyleTransfer.git
   cd WebStyleTransfer
   ```
2. **Install local dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Modal authentication**
   ```bash
   modal setup
   ```
4. **Deploy the backend to Modal**
    ```bash
    modal deploy modal_backend.py
    ```
5. **Push the frontend to GitHub**
    Upload `app.py` and `requirements.txt` to a new public or private repository on GitHub.
6. **Deploy the App**
    Go to [Streamlit Community Cloud](https://share.streamlit.io/), link your GitHub, and select your repository to deploy.
7. **Connect the Secrets**
    * Open your File Explorer and navigate to `C:\Users\[Your Username]\`.
    * Look for a file named `.modal.toml`.
    * Open this file in any text editor and look for
    ```toml
    [default]
    token_id = "ak-..."
    token_secret = "as-..."
    ```
    Copy them.
    * Go to your Streamlit dashboard, click the three dots next to your app, and select **Settings > Secrets**.
    * Paste the following into the text box, replacing the placeholders with your actual keys
    ```toml
    MODAL_TOKEN_ID = "ak-YOUR_TOKEN_ID"
    MODAL_TOKEN_SECRET = "as-YOUR_TOKEN_SECRET"
    ```

## References and Acknowledgements

This project builds upon the mathematical foundations and open-source code provided by the following researchers and developers:

* **Style Transfer (STROTSS):** Based on the paper *Style Transfer by Relaxed Optimal Transport and Self-Similarity* by Kolkin et al. [https://arxiv.org/abs/1904.12785](https://arxiv.org/abs/1904.12785). The original algorithm implementation is available at [https://github.com/nkolkin13/STROTSS](https://github.com/nkolkin13/STROTSS), and the PyTorch implementation utilized for this project is based on [https://github.com/futscdav/strotss](https://github.com/futscdav/strotss).
* **Sliced Color Transfer:** The sliced optimal transport implementation was adapted from the [STORIMAGING](https://storimaging.github.io/) Contrast and Color Notebooks ([https://storimaging.github.io/notebooksContrastAndColor/](https://storimaging.github.io/notebooksContrastAndColor/)). This educational resource and its algorithms are developed and maintained by [Julie Delon](https://judelo.github.io/), [Bruno Galerne](https://www.idpoisson.fr/galerne/), [Agnès Desolneux](https://desolneux.perso.math.cnrs.fr/), [Valentin De Bortoli](https://vdeborto.github.io/), and [Lucía Bouza](https://www.linkedin.com/in/lucia-bouza-heguerte/).