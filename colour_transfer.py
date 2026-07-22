import ot
import numpy as np
from time import time
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from argparse import ArgumentParser


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_image(path):
    """Read an image as float RGB in [0, 1], dropping any alpha channel.

    plt.imread returns floats in [0, 1] for PNGs but uint8 [0, 255] for JPEGs,
    so we normalise defensively.
    """
    im = plt.imread(path)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    return im[..., :3]


def uniform_weights(n):
    return np.ones(n) / n


# ----------------------------------------------------------------------
# Exact / entropic OT on a reduced colour histogram
# ----------------------------------------------------------------------
def transport_plan(X, Y, metric="sqeuclidean"):
    a, b = uniform_weights(len(X)), uniform_weights(len(Y))
    M = ot.dist(X, Y, metric)
    gamma = ot.emd(a, b, M)
    return gamma, M


def assignment_from_plan(gamma, Y):
    return Y[np.argmax(gamma, axis=1)]


def color_cloud_transfer_full(source, target, reg=1e-3, max_iter=10000, n_bins=24):
    X = source.reshape((-1, 3))
    Y = target.reshape((-1, 3))

    Xq = np.round(X * (n_bins - 1)).astype(int)
    Yq = np.round(Y * (n_bins - 1)).astype(int)

    x_vals, x_counts = np.unique(Xq, axis=0, return_counts=True)
    y_vals, y_counts = np.unique(Yq, axis=0, return_counts=True)

    a = x_counts / x_counts.sum()
    b = y_counts / y_counts.sum()

    Xb = x_vals / (n_bins - 1)
    Yb = y_vals / (n_bins - 1)

    M = ot.dist(Xb, Yb, metric="sqeuclidean")
    gamma = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)

    bin_to_color = {}
    for i in range(len(Xb)):
        j = np.argmax(gamma[i])
        bin_to_color[tuple(x_vals[i])] = Yb[j]

    out_pixels = np.empty_like(X)
    for i, xb in enumerate(Xq):
        out_pixels[i] = bin_to_color.get(tuple(xb), Yb[np.argmax(b)])

    out = np.clip(out_pixels.reshape(source.shape[:-1] + (3,)), 0, 1)
    return out, gamma


# ----------------------------------------------------------------------
# Sliced OT
# ----------------------------------------------------------------------
def color_cloud_transfer_sliced(source, target, n_iter=40, step=1.0, seed=0):
    """Transfer target's colour distribution onto source via sliced OT.
    """
    X = np.clip(source.reshape(-1, 3).astype(np.float64), 0, 1)
    Y = np.clip(target.reshape(-1, 3).astype(np.float64), 0, 1)

    rng = np.random.RandomState(seed)
    Z = X.copy()
    n, m = Z.shape[0], Y.shape[0]

    # quantile positions of the source pixels, mapped to target-sorted indices
    q = (np.arange(n) + 0.5) / n
    idx = np.clip((q * m).astype(int), 0, m - 1)

    for _ in range(n_iter):
        basis, _ = np.linalg.qr(rng.randn(3, 3))      # random orthonormal basis
        for k in range(3):
            theta = basis[:, k]
            Zp = Z @ theta
            Yp_sorted = np.sort(Y @ theta)
            order = np.argsort(Zp)                     # ranks of source projections

            target_proj = np.empty(n)
            target_proj[order] = Yp_sorted[idx]        # 1-D OT match by quantile

            # move only along theta by the projected displacement
            Z += step * ((target_proj - Zp)[:, None]) * theta[None, :]

    out = np.clip(Z, 0, 1).reshape(source.shape)
    return out, None


# ----------------------------------------------------------------------
# Regularisation:  guided-filter of the difference map  (He et al., ECCV 2010)
# ----------------------------------------------------------------------
def _box_mean(a, r):
    """Mean over a (2r+1)x(2r+1) window; 'reflect' avoids darkening at borders."""
    return uniform_filter(a, size=2 * r + 1, mode="reflect")


def guided_filter(p, guide, r=20, eps=1e-4):
    """Edge-preserving filter of image p, guided by `guide` (both 2-D, [0,1])."""
    mean_I = _box_mean(guide, r)
    mean_p = _box_mean(p, r)
    mean_Ip = _box_mean(guide * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_mean(guide * guide, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_mean(a, r)
    mean_b = _box_mean(b, r)
    return mean_a * guide + mean_b


def regularize(source, transferred, r=20, eps=1e-4):
    """Reduce colour-transfer artifacts by filtering the difference map.

    We compute M = transferred - source, smooth each channel of M with the
    guided filter (guided by the corresponding source channel so edges stay
    aligned to the ORIGINAL image), then rebuild  out = source + filtered(M).
    This removes JPEG blocks / noise amplification while preserving detail.
    """
    source = source[..., :3].astype(np.float64)
    transferred = transferred[..., :3].astype(np.float64)
    diff = transferred - source

    out = np.zeros_like(source)
    for c in range(3):
        out[:, :, c] = source[:, :, c] + guided_filter(diff[:, :, c],
                                                        source[:, :, c], r, eps)
    return np.clip(out, 0, 1)


def to_uint8(img):
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("--output", type=str, default="transferred.png")
    parser.add_argument("--type", type=str, default="sliced",
                        choices=["full", "sliced"])
    parser.add_argument("--iters", type=int, default=40,
                        help="sliced-OT iterations (each uses 3 directions)")
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--regularize", action="store_true",
                        help="apply guided-filter regularisation to the result")
    parser.add_argument("--radius", type=int, default=20,
                        help="guided-filter window radius")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="guided-filter regularisation")
    args = parser.parse_args()

    A1 = load_image(args.source)
    C1 = load_image(args.target)

    start = time()
    if args.type == "full":
        out, _ = color_cloud_transfer_full(A1, C1, reg=1e-2, max_iter=5000, n_bins=48)
    else:
        out, _ = color_cloud_transfer_sliced(A1, C1, n_iter=args.iters, step=args.step)

    if args.regularize:
        out = regularize(A1, out, r=args.radius, eps=args.eps)

    Image.fromarray(to_uint8(out)).save(args.output)
    print(f"Done in {time() - start:.3f}s")