"""Image kernels behind the image ideas: the Rust ``sdx_rs`` build when present (``cargo build --release`` in
``native/rust``), NumPy otherwise. All take and return float32 ``(H, W, C)`` arrays in [0, 1]."""

from __future__ import annotations

import functools

import numpy as np

import utils.native  # noqa: F401  (puts sdx_native on sys.path)
from sdx_native import ffi


def error_diffuse(img: np.ndarray, levels: int = 256, amount: float = 1.0) -> np.ndarray | None:
    """Floyd-Steinberg quantisation, or ``None`` without the Rust build (a Python loop takes seconds per image)."""
    return ffi.maybe_error_diffuse_rust(img, levels, amount)


@functools.cache
def blue_noise(n: int = 64, sigma: float = 1.5, seed: int = 0) -> np.ndarray:
    """``(n, n)`` void-and-cluster blue-noise threshold map in (0, 1) (cached; ~0.2 s in NumPy, ~5 ms in Rust)."""
    out = ffi.maybe_blue_noise_rust(n, sigma, seed)
    return out if out is not None else _void_and_cluster(n, sigma, seed)


def _void_and_cluster(n: int, sigma: float, seed: int) -> np.ndarray:
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(np.float32)
    kern = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma**2)).astype(np.float32)
    size = n * n

    def splat(e: np.ndarray, p: int, sign: float) -> None:
        e += sign * np.roll(kern, (p // n, p % n), axis=(0, 1)).ravel()

    def cluster(pat: np.ndarray, e: np.ndarray) -> int:
        return int(np.where(pat, e, -np.inf).argmax())

    def void(pat: np.ndarray, e: np.ndarray) -> int:
        return int(np.where(pat, np.inf, e).argmin())

    pattern = np.zeros(size, bool)
    energy = np.zeros(size, np.float32)
    for p in np.random.default_rng(seed).choice(size, max(1, size // 10), replace=False):
        pattern[p] = True
        splat(energy, int(p), 1.0)
    for _ in range(size):
        c = cluster(pattern, energy)
        pattern[c] = False
        splat(energy, c, -1.0)
        v = void(pattern, energy)
        pattern[v] = True
        splat(energy, v, 1.0)
        if v == c:
            break
    ones = int(pattern.sum())
    rank = np.zeros(size, np.int64)
    proto, e = pattern.copy(), energy.copy()
    for r in range(ones - 1, -1, -1):
        p = cluster(proto, e)
        proto[p] = False
        splat(e, p, -1.0)
        rank[p] = r
    proto, e = pattern.copy(), energy.copy()
    for r in range(ones, size):
        p = void(proto, e)
        proto[p] = True
        splat(e, p, 1.0)
        rank[p] = r
    return ((rank + 0.5) / size).astype(np.float32).reshape(n, n)


def bilateral(img: np.ndarray, radius: int = 3, sigma_s: float = 2.0, sigma_r: float = 0.08) -> np.ndarray:
    out = ffi.maybe_bilateral_rust(img, radius, sigma_s, sigma_r)
    if out is not None:
        return out
    img = np.ascontiguousarray(img, dtype=np.float32)
    h, w, _ = img.shape
    pad = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    acc = np.zeros_like(img)
    norm = np.zeros((h, w, 1), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = pad[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            d2 = ((nb - img) ** 2).sum(-1, keepdims=True)
            wgt = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s**2) - d2 / (2 * sigma_r**2))
            acc += wgt * nb
            norm += wgt
    return acc / norm


def median3(img: np.ndarray) -> np.ndarray:
    out = ffi.maybe_median3_rust(img)
    if out is not None:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    img = np.asarray(img, dtype=np.float32)
    h, w, c = img.shape
    win = sliding_window_view(np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge"), (3, 3), axis=(0, 1))
    return np.median(win.reshape(h, w, c, 9), axis=-1).astype(np.float32)


def kuwahara(img: np.ndarray, radius: int = 3) -> np.ndarray:
    out = ffi.maybe_kuwahara_rust(img, radius)
    if out is not None:
        return out
    img = np.asarray(img, dtype=np.float32)
    h, w, c = img.shape
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32) if c >= 3 else img[..., 0]
    planes = np.concatenate([img, lum[..., None], (lum * lum)[..., None]], -1).astype(np.float64)
    sat = np.zeros((h + 1, w + 1, c + 2))
    sat[1:, 1:] = planes.cumsum(0).cumsum(1)
    ys, xs = np.mgrid[0:h, 0:w]
    ylo, yhi = np.maximum(ys - radius, 0), np.minimum(ys + radius, h - 1)
    xlo, xhi = np.maximum(xs - radius, 0), np.minimum(xs + radius, w - 1)
    means, variances = [], []
    for y0, x0, y1, x1 in ((ylo, xlo, ys, xs), (ylo, xs, ys, xhi), (ys, xlo, yhi, xs), (ys, xs, yhi, xhi)):
        s = sat[y1 + 1, x1 + 1] - sat[y0, x1 + 1] - sat[y1 + 1, x0] + sat[y0, x0]
        m = s / ((y1 - y0 + 1) * (x1 - x0 + 1))[..., None]
        means.append(m[..., :c])
        variances.append(m[..., c + 1] - m[..., c] ** 2)
    pick = np.argmin(np.stack(variances), axis=0)
    return np.take_along_axis(np.stack(means), pick[None, ..., None], 0)[0].astype(np.float32)
