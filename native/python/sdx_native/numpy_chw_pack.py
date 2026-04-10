"""HWC uint8 / float arrays → CHW float32 (dataset / VAE preprocess helpers)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def hwc_u8_to_chw_f32(img: np.ndarray, *, scale: str = "01") -> np.ndarray:
    """
    ``img`` shape ``(H, W, C)`` uint8 or float. Returns ``(C, H, W)`` float32.

    ``scale``:
      - ``"01"``: divide uint8 by 255
      - ``"n11"``: ``(x/255)*2-1``
    """
    if img.ndim != 3:
        raise ValueError("expected HWC array")
    if img.dtype == np.uint8:
        x = img.astype(np.float32) / 255.0
    else:
        x = img.astype(np.float32, copy=False)
    if scale == "n11":
        if img.dtype != np.uint8:
            x = np.clip(x, 0.0, 1.0)
        x = x * 2.0 - 1.0
    elif scale == "01":
        pass
    elif scale == "raw":
        pass
    else:
        raise ValueError("scale must be 01, n11, or raw")
    return np.transpose(x, (2, 0, 1)).copy()


def chw_f32_to_hwc_u8(img: np.ndarray, *, from_scale: str = "01") -> np.ndarray:
    """``(C,H,W)`` float → ``(H,W,C)`` uint8."""
    if img.ndim != 3:
        raise ValueError("expected CHW array")
    x = np.transpose(img, (1, 2, 0))
    if from_scale == "n11":
        x = (x + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def stack_chw_batch(images_chw: np.ndarray) -> np.ndarray:
    """``(N, C, H, W)`` ensure contiguous float32."""
    if images_chw.ndim != 4:
        raise ValueError("expected NCHW")
    return np.ascontiguousarray(images_chw.astype(np.float32, copy=False))


def channel_mean_std(chw: np.ndarray, *, axis_spatial: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over spatial dims (batch NCHW: pass single image or reduce batch)."""
    if chw.ndim == 3:
        m = chw.mean(axis=axis_spatial)
        s = chw.std(axis=axis_spatial)
        return m, s
    if chw.ndim == 4:
        m = chw.mean(axis=(0, 2, 3))
        s = chw.std(axis=(0, 2, 3))
        return m, s
    raise ValueError("expected CHW or NCHW")
