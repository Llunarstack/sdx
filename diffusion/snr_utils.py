"""
SNR / alpha_cumprod helpers for schedule inspection and tooling.

Uses the Rust ``sdx_diffusion_math`` cdylib when built (no NumPy overhead,
no GIL contention). Falls back to pure NumPy otherwise.

Training uses torch tensors inside ``GaussianDiffusion``; these functions
are for analysis, monitoring, and schedule comparison without a GPU.
"""

from __future__ import annotations

import numpy as np

__all__ = ["alpha_cumprod_from_betas", "snr_from_alpha_cumprod", "snr_from_betas"]


def alpha_cumprod_from_betas(betas: np.ndarray) -> np.ndarray:
    """``alpha_cumprod[t] = prod_{s<=t} (1 - beta[s])``."""
    try:
        from sdx_native.diffusion_math_native import maybe_alpha_cumprod_rust
        result = maybe_alpha_cumprod_rust(np.asarray(betas, dtype=np.float64))
        if result is not None:
            return result
    except Exception:
        pass
    beta = np.asarray(betas, dtype=np.float64)
    return np.cumprod(1.0 - beta)


def snr_from_alpha_cumprod(alpha_cumprod: np.ndarray) -> np.ndarray:
    """``SNR(t) = alpha_bar[t] / (1 - alpha_bar[t])`` for VP diffusion."""
    try:
        from sdx_native.diffusion_math_native import maybe_snr_from_alpha_cumprod_rust
        result = maybe_snr_from_alpha_cumprod_rust(np.asarray(alpha_cumprod, dtype=np.float64))
        if result is not None:
            return result
    except Exception:
        pass
    try:
        from sdx_native.diffusion_sigma_fast import snr_from_alpha_cumprod as snr_from_alpha_bar_numpy

        return snr_from_alpha_bar_numpy(np.asarray(alpha_cumprod, dtype=np.float64))
    except Exception:
        pass
    ab = np.asarray(alpha_cumprod, dtype=np.float64)
    return ab / (1.0 - ab + 1e-8)


def snr_from_betas(betas: np.ndarray) -> np.ndarray:
    """SNR curve from a raw beta schedule vector."""
    return snr_from_alpha_cumprod(alpha_cumprod_from_betas(betas))
