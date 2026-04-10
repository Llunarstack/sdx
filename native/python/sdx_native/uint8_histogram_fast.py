"""Per-channel histograms for uint8 HWC (dataset QA, clipping stats)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def histogram_u8_channels(hwc: np.ndarray) -> List[np.ndarray]:
    """
    ``hwc`` uint8, shape ``(H,W,C)``. Returns list of length ``C`` with shape ``(256,)`` int64 counts.
    """
    if hwc.ndim != 3:
        raise ValueError("expected HWC uint8")
    if hwc.dtype != np.uint8:
        hwc = hwc.astype(np.uint8, copy=False)
    c = hwc.shape[2]
    out: List[np.ndarray] = []
    for i in range(c):
        out.append(np.bincount(hwc[:, :, i].reshape(-1), minlength=256).astype(np.int64))
    return out


def luminance_histogram_u8(hwc: np.ndarray) -> np.ndarray:
    """BT.601 luma per pixel, 256-bin histogram."""
    if hwc.ndim != 3 or hwc.shape[2] < 3:
        raise ValueError("need HWC with >=3 channels")
    r = hwc[:, :, 0].astype(np.int32)
    g = hwc[:, :, 1].astype(np.int32)
    b = hwc[:, :, 2].astype(np.int32)
    y = ((77 * r + 150 * g + 29 * b + 128) >> 8).clip(0, 255).astype(np.uint8)
    return np.bincount(y.reshape(-1), minlength=256).astype(np.int64)


def clip_histogram_percentiles(hist: np.ndarray, *, low_pct: float, high_pct: float) -> Tuple[int, int]:
    """Inclusive bin indices from cumulative distribution (0..255)."""
    h = hist.astype(np.float64)
    total = h.sum()
    if total <= 0:
        return 0, 255
    cdf = np.cumsum(h) / total
    low = int(np.searchsorted(cdf, low_pct))
    high = int(np.searchsorted(cdf, high_pct))
    return max(0, low), min(255, high)
