"""Numpy helpers for latent tensors (crop, pad, MSE) — CPU fast paths."""

from __future__ import annotations

import numpy as np


def center_crop_hw(x: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    """Crop last two dims (``... H W``) to ``target_h`` x ``target_w`` from center."""
    if x.ndim < 2:
        raise ValueError("need at least HW")
    h, w = x.shape[-2], x.shape[-1]
    if target_h > h or target_w > w:
        raise ValueError("target larger than tensor")
    t0 = (h - target_h) // 2
    l0 = (w - target_w) // 2
    return x[..., t0 : t0 + target_h, l0 : l0 + target_w].copy()


def reflect_pad_hw(x: np.ndarray, *, pad_h: int, pad_w: int) -> np.ndarray:
    """Pad ``... H W`` symmetrically with edge reflect (numpy pad modes)."""
    if pad_h < 0 or pad_w < 0:
        raise ValueError("pad must be non-negative")
    if pad_h == 0 and pad_w == 0:
        return x
    pads = [(0, 0)] * (x.ndim - 2) + [(pad_h, pad_h), (pad_w, pad_w)]
    return np.pad(x, pads, mode="reflect")


def latent_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error (scalar)."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def latent_flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity treating flattened vectors."""
    fa = a.reshape(-1).astype(np.float64)
    fb = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(fa)
    nb = np.linalg.norm(fb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(fa, fb) / (na * nb))


def batch_latent_rms(x: np.ndarray) -> np.ndarray:
    """RMS per batch item for ``(N, ...)`` tensor."""
    if x.ndim < 2:
        raise ValueError("expected batch dim")
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    return np.sqrt(np.mean(flat * flat, axis=1)).astype(np.float32)
