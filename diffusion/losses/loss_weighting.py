"""Optional per-timestep loss-weighting strategies (ported from generative-models).

The diffusion loss is averaged over randomly sampled timesteps, but not all
timesteps are equally informative: very-high-noise steps carry almost no signal,
very-low-noise steps are nearly trivial. A weighting function rescales each step's
loss to rebalance where the model spends its capacity, which can noticeably change
sample quality and convergence.

These are used when ``loss_weighting`` is not ``"min_snr"`` in config (min-SNR is
applied separately inside ``training_losses``). Everything keys off the noise level
``sigma = sqrt(1 - alpha_cumprod)`` for each timestep.
"""

import torch


def sigma_from_alpha_cumprod(alpha_cumprod: torch.Tensor) -> torch.Tensor:
    """Noise level sigma from alpha_cumprod (variance of noise in x_t is 1 - alpha_cumprod)."""
    return (1.0 - alpha_cumprod).clamp(min=1e-8).sqrt()


def unit_weight(sigma: torch.Tensor) -> torch.Tensor:
    """No reweighting — every timestep contributes equally (the neutral baseline)."""
    return torch.ones_like(sigma)


def edm_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    """EDM weighting (Karras et al.): (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.

    Emphasizes the mid-noise region where most perceptually important learning
    happens. ``sigma_data`` is the assumed std of the clean data (0.5 for typical
    VAE latents).
    """
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data).clamp(min=1e-8) ** 2


def v_weight(sigma: torch.Tensor) -> torch.Tensor:
    """Weighting consistent with v-prediction — EDM with ``sigma_data = 1``."""
    return edm_weight(sigma, sigma_data=1.0)


def eps_weight(sigma: torch.Tensor) -> torch.Tensor:
    """Weighting for epsilon-prediction: ``sigma^-2`` (up-weights low-noise steps)."""
    return sigma ** (-2.0)


def get_loss_weight(
    alpha_cumprod: torch.Tensor,
    weighting: str,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    Per-timestep loss weight from alpha_cumprod.
    weighting: "unit" | "edm" | "v" | "eps". (min_snr is applied separately in training_losses.)
    """
    sigma = sigma_from_alpha_cumprod(alpha_cumprod)
    if weighting == "unit":
        return unit_weight(sigma)
    if weighting == "edm":
        return edm_weight(sigma, sigma_data=sigma_data)
    if weighting == "v":
        return v_weight(sigma)
    if weighting == "eps":
        return eps_weight(sigma)
    return unit_weight(sigma)
