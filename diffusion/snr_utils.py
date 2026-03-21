"""
NumPy helpers for **SNR / alpha** inspection (monitoring, ablations, tooling).

Training uses torch tensors inside ``GaussianDiffusion``; these functions help
analyze schedules from raw ``beta`` vectors or compare curves without a GPU.
"""

from __future__ import annotations

import numpy as np

__all__ = ["alpha_cumprod_from_betas", "snr_from_alpha_cumprod", "snr_from_betas"]


def alpha_cumprod_from_betas(betas: np.ndarray) -> np.ndarray:
    """``alpha_cumprod[t] = prod_{s<=t} (1 - beta_s)``."""
    beta = np.asarray(betas, dtype=np.float64)
    alpha = 1.0 - beta
    return np.cumprod(alpha)


def snr_from_alpha_cumprod(alpha_cumprod: np.ndarray) -> np.ndarray:
    """SNR(t) = alpha_bar / (1 - alpha_bar) for VP diffusion."""
    ab = np.asarray(alpha_cumprod, dtype=np.float64)
    return ab / (1.0 - ab + 1e-8)


def snr_from_betas(betas: np.ndarray) -> np.ndarray:
    """SNR curve from a beta schedule vector."""
    return snr_from_alpha_cumprod(alpha_cumprod_from_betas(betas))
