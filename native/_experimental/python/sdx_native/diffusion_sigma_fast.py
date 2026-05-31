"""Lightweight numpy SNR / sigma helpers (complements Rust ``sdx-diffusion-math``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def snr_from_alpha_cumprod(alpha_bar: NDArray[Any], *, denom_eps: float = 1e-8) -> NDArray[Any]:
    """SNR = alpha_bar / (1 - alpha_bar + denom_eps), matching Rust ``sdx_snr_from_alpha_cumprod_f64``."""
    a = np.asarray(alpha_bar, dtype=np.float64)
    return a / (1.0 - a + denom_eps)


def sigma_from_alpha_cumprod(alpha_bar: NDArray[Any], *, eps: float = 1e-20) -> NDArray[Any]:
    """DDPM-style sigma ~ sqrt((1-a)/a)."""
    a = np.clip(alpha_bar.astype(np.float64), eps, 1.0 - eps)
    return np.sqrt((1.0 - a) / a)


def clip_snr(snr: NDArray[Any], *, min_snr: float, max_snr: float) -> NDArray[Any]:
    return np.clip(snr.astype(np.float64), min_snr, max_snr)


def effective_noise_weight(snr: NDArray[Any], *, min_snr: float = 1e-4, max_snr: float = 1e4) -> NDArray[Any]:
    """``1/sqrt(SNR)`` clipped — useful for loss reweight sketches."""
    s = clip_snr(snr, min_snr=min_snr, max_snr=max_snr)
    return 1.0 / np.sqrt(s)
