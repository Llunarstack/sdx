"""
Unified **per-timestep loss weights** for VP diffusion training.

- ``min_snr`` — hard cap on SNR in the weight (SD-style).
- ``min_snr_soft`` — smooth alternative: ``gamma / (snr + gamma)`` (no hard ``clamp``);
  often slightly gentler on very low-SNR steps while still down-weighting high-SNR steps.
- ``unit`` | ``edm`` | ``v`` | ``eps`` — delegating to :mod:`diffusion.losses.loss_weighting`.

Use :func:`get_timestep_loss_weight` from both ``GaussianDiffusion.training_losses`` and
any parallel loss paths (e.g. MDM in ``train.py``) so behavior stays consistent.
"""

from __future__ import annotations

import torch

from .loss_weighting import get_loss_weight as _get_loss_weight

__all__ = ["get_timestep_loss_weight"]


def get_timestep_loss_weight(
    loss_weighting: str,
    *,
    snr: torch.Tensor | None,
    alpha_cumprod: torch.Tensor,
    min_snr_gamma: float,
    loss_weighting_sigma_data: float,
) -> torch.Tensor:
    """
    Per-batch-element weight ``(B,)`` multiplying the per-sample MSE.

    ``snr`` should be ``alpha_cumprod / (1 - alpha_cumprod)`` at the sampled timesteps, or
    ``None`` when only alpha-based weighting is used.
    """
    device = alpha_cumprod.device
    dtype = alpha_cumprod.dtype
    b = alpha_cumprod.shape[0]
    lw = str(loss_weighting).lower().strip()

    if lw == "min_snr":
        if min_snr_gamma > 0 and snr is not None:
            return torch.clamp(snr, max=min_snr_gamma) / (snr + 1e-8)
        return torch.ones(b, device=device, dtype=dtype)

    if lw == "min_snr_soft":
        if min_snr_gamma > 0 and snr is not None:
            g = float(min_snr_gamma)
            return g / (snr + g + 1e-8)
        return torch.ones(b, device=device, dtype=dtype)

    return _get_loss_weight(alpha_cumprod, loss_weighting, loss_weighting_sigma_data)
