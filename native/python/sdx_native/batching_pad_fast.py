"""Variable-length sequence padding (numpy) for collate helpers."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def pad_1d_sequences(
    sequences: Sequence[np.ndarray],
    *,
    pad_value: float = 0.0,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad list of 1D arrays to ``(B, T_max)``. Returns ``(batch, lengths)`` where lengths is ``(B,)`` int64.
    """
    if not sequences:
        raise ValueError("empty sequences")
    lens = np.array([len(s) for s in sequences], dtype=np.int64)
    tmax = int(lens.max())
    b = len(sequences)
    out = np.full((b, tmax), pad_value, dtype=dtype)
    for i, s in enumerate(sequences):
        sl = np.asarray(s, dtype=dtype)
        out[i, : sl.shape[0]] = sl
    return out, lens


def pad_2d_hw(images: Sequence[np.ndarray], *, pad_value: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad ``(H,W,C)`` images to common ``Hmax,Wmax`` (bottom-right align)."""
    if not images:
        raise ValueError("empty")
    hs = [x.shape[0] for x in images]
    ws = [x.shape[1] for x in images]
    c = images[0].shape[2]
    hm, wm = max(hs), max(ws)
    b = len(images)
    out = np.full((b, hm, wm, c), pad_value, dtype=images[0].dtype)
    for i, im in enumerate(images):
        h, w, _ = im.shape
        out[i, :h, :w, :] = im
    return out, (hm, wm)
