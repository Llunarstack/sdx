"""2D coordinate grids for positional encodings / flow (numpy, vectorized)."""

from __future__ import annotations

import numpy as np


def normalized_grid(h: int, w: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    ``(H, W, 2)`` with x in ``[-1,1]`` (width), y in ``[-1,1]`` (height), corners at ±1.
    """
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float64)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.stack([xx, yy], axis=-1)
    return g.astype(dtype, copy=False)


def pixel_center_grid(h: int, w: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """``(H,W,2)`` integer pixel centers ``(x,y)`` as ``0.5 + arange``."""
    xs = np.arange(w, dtype=np.float64) + 0.5
    ys = np.arange(h, dtype=np.float64) + 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx, yy], axis=-1).astype(dtype, copy=False)
