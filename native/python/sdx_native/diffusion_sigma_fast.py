"""Lightweight numpy SNR / sigma helpers (complements Rust ``sdx-diffusion-math``)."""

from __future__ import annotations

import numpy as np


def snr_from_alpha_cumprod(alpha_bar: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    """SNR = alpha_bar / (1 - alpha_bar) for each timestep."""
    a = np.clip(alpha_bar.astype(np.float64), eps, 1.0 - eps)
    return a / (1.0 - a)


def sigma_from_alpha_cumprod(alpha_bar: np.ndarray, *, eps: float = 1e-20) -> np.ndarray:
    """DDPM-style sigma ~ sqrt((1-a)/a)."""
    a = np.clip(alpha_bar.astype(np.float64), eps, 1.0 - eps)
    return np.sqrt((1.0 - a) / a)


def clip_snr(snr: np.ndarray, *, min_snr: float, max_snr: float) -> np.ndarray:
    return np.clip(snr.astype(np.float64), min_snr, max_snr)


def effective_noise_weight(snr: np.ndarray, *, min_snr: float = 1e-4, max_snr: float = 1e4) -> np.ndarray:
    """``1/sqrt(SNR)`` clipped — useful for loss reweight sketches."""
    s = clip_snr(snr, min_snr=min_snr, max_snr=max_snr)
    return 1.0 / np.sqrt(s)
