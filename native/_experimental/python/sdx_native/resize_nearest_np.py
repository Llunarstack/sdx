"""Nearest-neighbor resize for uint8/float HWC (no PIL — batch-friendly)."""

from __future__ import annotations

import numpy as np


def resize_hwc_nearest(img: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    """Map source integer coords via rounding; ``img`` is ``(H,W,C)``."""
    if img.ndim != 3:
        raise ValueError("HWC only")
    h, w, c = img.shape
    if out_h < 1 or out_w < 1:
        raise ValueError("bad output size")
    ys = np.clip(np.round(np.linspace(0, h - 1, out_h)).astype(np.int64), 0, h - 1)
    xs = np.clip(np.round(np.linspace(0, w - 1, out_w)).astype(np.int64), 0, w - 1)
    return img[np.ix_(ys, xs, np.arange(c))].copy()


def downscale_max_hwc(img: np.ndarray, *, max_side: int) -> np.ndarray:
    """Preserve aspect; longest side becomes ``max_side`` (nearest)."""
    h, w = img.shape[0], img.shape[1]
    m = max(h, w)
    if m <= max_side:
        return img.copy()
    scale = max_side / float(m)
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    return resize_hwc_nearest(img, out_h=out_h, out_w=out_w)
