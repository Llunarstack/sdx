"""High-level AI acceleration facade (Python + optional Rust/C/C++/CUDA).

Import from here in product code so call sites stay stable whether or not
native libraries are built:

    from utils.ai_accel import canny_control_map, style_subtract, cfg_combine

Each helper returns a NumPy/PIL result using native code when available,
otherwise a pure-Python/NumPy fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure ``sdx_native`` resolves outside pytest (same bootstrap as utils.native).
_SDX_NATIVE_PY = Path(__file__).resolve().parents[2] / "native" / "_experimental" / "python"
if str(_SDX_NATIVE_PY) not in sys.path:
    sys.path.insert(0, str(_SDX_NATIVE_PY))

__all__ = [
    "canny_control_map",
    "cfg_combine",
    "native_stack_summary",
    "quality_highlight_frac",
    "quality_laplacian_var",
    "quality_midtone_entropy",
    "style_l2_normalize",
    "style_subtract",
    "style_weighted_mean",
]


def canny_control_map(
    rgb: np.ndarray,
    *,
    low: float = 40.0,
    high: float = 100.0,
    soft: bool = False,
) -> np.ndarray:
    """Grayscale uint8 edges; Rust Canny if built, else PIL FIND_EDGES proxy."""
    try:
        from sdx_native.canny_ops_native import canny_u8

        hit = canny_u8(rgb, low=low, high=high, soft=soft)
        if hit is not None:
            return hit
    except Exception:
        pass
    from PIL import Image, ImageFilter

    arr = np.asarray(rgb)
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(arr[..., :3].astype(np.uint8), mode="RGB").convert("L")
    if soft:
        out = img.filter(ImageFilter.GaussianBlur(radius=1)).filter(ImageFilter.FIND_EDGES)
    else:
        out = img.filter(ImageFilter.FIND_EDGES)
    return np.asarray(out, dtype=np.uint8)


def style_weighted_mean(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    try:
        from sdx_native.style_embed_native import weighted_mean

        hit = weighted_mean(rows, weights)
        if hit is not None:
            return hit
    except Exception:
        pass
    mat = np.asarray(rows, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 1e-12:
        return np.zeros((mat.shape[-1],), dtype=np.float32)
    w = w / s
    return (mat * w[:, None]).sum(axis=0).astype(np.float32)


def style_subtract(image: np.ndarray, content: np.ndarray, strength: float = 1.0) -> np.ndarray:
    try:
        from sdx_native.style_embed_native import subtract

        hit = subtract(image, content, strength=strength)
        if hit is not None:
            return hit
    except Exception:
        pass
    a = np.asarray(image, dtype=np.float32).reshape(-1)
    b = np.asarray(content, dtype=np.float32).reshape(-1)
    return (a - float(strength) * b).astype(np.float32)


def style_l2_normalize(rows: np.ndarray) -> np.ndarray:
    try:
        from sdx_native.style_embed_native import l2_normalize_rows

        hit = l2_normalize_rows(rows)
        if hit is not None:
            return hit
    except Exception:
        pass
    mat = np.asarray(rows, dtype=np.float32)
    flat = mat.ndim == 1
    if flat:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    out = (mat / norms).astype(np.float32)
    return out.reshape(-1) if flat else out


def cfg_combine(
    cond: np.ndarray,
    uncond: np.ndarray,
    *,
    scale: float = 7.5,
    rescale_phi: float = 0.0,
) -> np.ndarray:
    try:
        from sdx_native.cfg_combine_native import cfg_combine_numpy

        return cfg_combine_numpy(cond, uncond, scale=scale, rescale_phi=rescale_phi)
    except Exception:
        a = np.asarray(cond, dtype=np.float32)
        b = np.asarray(uncond, dtype=np.float32)
        out = b + float(scale) * (a - b)
        phi = float(rescale_phi)
        if phi <= 0.0:
            return out
        std_c = float(np.std(a)) + 1e-8
        std_o = float(np.std(out)) + 1e-8
        mean_o = float(np.mean(out))
        centered = (out - mean_o) * (std_c / std_o) + mean_o
        return (phi * centered + (1.0 - phi) * out).astype(np.float32)


def quality_laplacian_var(rgb: np.ndarray) -> float:
    try:
        from sdx_native.quality_scorers_native import laplacian_var

        hit = laplacian_var(rgb)
        if hit is not None:
            return float(hit)
    except Exception:
        pass
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    g = gray.astype(np.float64)
    lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var()) if lap.size else 0.0


def quality_highlight_frac(rgb: np.ndarray, thr: float = 245.0) -> float:
    try:
        from sdx_native.quality_scorers_native import highlight_frac

        hit = highlight_frac(rgb, thr=thr)
        if hit is not None:
            return float(hit)
    except Exception:
        pass
    arr = np.asarray(rgb)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(np.mean(gray >= float(thr)))


def quality_midtone_entropy(rgb: np.ndarray, lo: float = 40.0, hi: float = 220.0) -> float:
    try:
        from sdx_native.quality_scorers_native import midtone_entropy

        hit = midtone_entropy(rgb, lo=lo, hi=hi)
        if hit is not None:
            return float(hit)
    except Exception:
        pass
    arr = np.asarray(rgb)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mid = gray[(gray >= lo) & (gray <= hi)]
    if mid.size < 8:
        return 0.0
    hist, _ = np.histogram(mid, bins=64, range=(lo, hi), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist + 1e-12)).sum())


def native_stack_summary() -> dict[str, Any]:
    """Which accel backends are loadable (for dashboards / copilot notes)."""
    out: dict[str, Any] = {}
    try:
        from sdx_native import canny_ops_native, cfg_combine_native, quality_scorers_native, style_embed_native

        out["canny"] = bool(canny_ops_native.available())
        out["style_embed"] = bool(style_embed_native.available())
        out["quality"] = bool(quality_scorers_native.available())
        out["cfg_combine"] = bool(cfg_combine_native.available())
    except Exception as exc:
        out["error"] = str(exc)
    try:
        from sdx_native.native_tools import native_stack_status

        out["paths"] = native_stack_status()
    except Exception:
        pass
    return out
