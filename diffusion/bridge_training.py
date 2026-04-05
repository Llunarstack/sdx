"""
Lightweight **latent-bridge** auxiliary loss: interpolate two latents in a batch, diffuse with VP
``q_sample``, and apply the usual denoising target (epsilon / v / x0) via ``GaussianDiffusion``.

This is **not** a full Schrödinger-bridge trainer; it is a regularizer that exposes the model to
``x_t`` lying between **different** training examples (shuffle pairing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from diffusion.gaussian_diffusion import GaussianDiffusion


def shuffle_pair_latents(latents: torch.Tensor) -> torch.Tensor:
    """Permute batch for a second endpoint (no fixed pairs with self)."""
    b = latents.shape[0]
    if b < 2:
        return latents.roll(1, dims=0)
    perm = torch.randperm(b, device=latents.device)
    while bool((perm == torch.arange(b, device=latents.device)).all()):
        perm = torch.randperm(b, device=latents.device)
    return latents[perm]


def bridge_aux_vp_loss(
    diffusion: "GaussianDiffusion",
    model: torch.nn.Module,
    latents: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict,
    *,
    mix_lambda: float = 0.15,
    noise: Optional[torch.Tensor] = None,
    noise_offset: float = 0.0,
    min_snr_gamma: float = 5.0,
    loss_weighting: str = "min_snr",
    loss_weighting_sigma_data: float = 0.5,
    use_spectral_sfp_loss: bool = False,
    spectral_sfp_low_sigma: float = 0.22,
    spectral_sfp_high_sigma: float = 0.22,
    spectral_sfp_tau_power: float = 1.0,
) -> torch.Tensor:
    """
    Endpoint ``x_a = latents``, ``x_b = shuffle_pair(latents)``, ``x0 = (1-λ) x_a + λ x_b``.
    Returns **scalar** VP training loss on ``x0`` (same API knobs as ``training_losses``).
    """
    lam = float(max(0.0, min(1.0, mix_lambda)))
    if lam <= 0.0:
        raise ValueError("mix_lambda must be > 0")
    xb = shuffle_pair_latents(latents)
    x0 = (1.0 - lam) * latents + lam * xb
    return diffusion.training_losses(
        model,
        x0,
        t,
        model_kwargs=model_kwargs,
        noise=noise,
        refinement_prob=0.0,
        refinement_max_t=0,
        noise_offset=noise_offset,
        min_snr_gamma=min_snr_gamma,
        sample_weights=None,
        loss_weighting=loss_weighting,
        loss_weighting_sigma_data=loss_weighting_sigma_data,
        use_spectral_sfp_loss=use_spectral_sfp_loss,
        spectral_sfp_low_sigma=spectral_sfp_low_sigma,
        spectral_sfp_high_sigma=spectral_sfp_high_sigma,
        spectral_sfp_tau_power=spectral_sfp_tau_power,
    )["loss"]


__all__ = ["bridge_aux_vp_loss", "shuffle_pair_latents"]
