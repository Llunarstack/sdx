"""Attention bias packing (bool/float masks → additive bias for transformers)."""

from __future__ import annotations

import numpy as np


def bool_mask_to_additive(
    mask: np.ndarray,
    *,
    true_value: float = 0.0,
    false_value: float = -1e4,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    ``mask`` bool ``(B, L)`` or ``(L,)`` — True = keep, False = block.
    Returns float mask suitable for softmax bias.
    """
    m = np.asarray(mask, dtype=bool)
    out = np.full(m.shape, false_value, dtype=dtype)
    out[m] = true_value
    return out


def causal_mask_1d(length: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Additive mask ``(L,L)``: 0 on/below main diagonal (can attend), large negative above."""
    neg = np.float32(-1e4) if dtype == np.float32 else np.float64(-1e4)
    return np.triu(np.ones((length, length), dtype=dtype) * neg, k=1)
