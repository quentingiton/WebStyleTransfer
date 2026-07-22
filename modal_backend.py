import io
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "scipy", "POT", "matplotlib", "pillow")
    .add_local_file("strotss.py", remote_path="/root/strotss.py")
    .add_local_file("colour_transfer.py", remote_path="/root/colour_transfer.py")
)
app = modal.App("image-transfer", image=image)

@app.function(gpu="T4", timeout=600)
def run_style_transfer(source_bytes: bytes, style_bytes: bytes, weight: float, resize_to: int) -> bytes:
    import strotss
    import PIL.Image

    source_pil = PIL.Image.open(io.BytesIO(source_bytes)).convert("RGB")
    style_pil = PIL.Image.open(io.BytesIO(style_bytes)).convert("RGB")

    source_resized = strotss.pil_resize_long_edge_to(source_pil, resize_to)
    style_resized = strotss.pil_resize_long_edge_to(style_pil, resize_to)
    result_pil = strotss.strotss(source_resized, style_resized, content_weight=weight*16.0, device="cuda:0", space="uniform")

    out_buf = io.BytesIO()
    result_pil.save(out_buf, format="PNG")
    return out_buf.getvalue()

@app.function()
def run_color_transfer(source_bytes: bytes, target_bytes: bytes, iters: int, step: float, apply_reg: bool) -> bytes:
    import colour_transfer
    import PIL.Image
    import numpy as np

    source_pil = PIL.Image.open(io.BytesIO(source_bytes)).convert("RGB")
    target_pil = PIL.Image.open(io.BytesIO(target_bytes)).convert("RGB")
    
    source_np = np.array(source_pil).astype(np.float32) / 255.0
    target_np = np.array(target_pil).astype(np.float32) / 255.0
    
    out_np, _ = colour_transfer.color_cloud_transfer_sliced(source_np, target_np, n_iter=iters, step=step)

    if apply_reg:
        out_np = colour_transfer.regularize(source_np, out_np, r=20, eps=1e-4)
    
    out_uint8 = colour_transfer.to_uint8(out_np)
    out_pil = PIL.Image.fromarray(out_uint8)
    
    out_buf = io.BytesIO()
    out_pil.save(out_buf, format="PNG")
    return out_buf.getvalue()