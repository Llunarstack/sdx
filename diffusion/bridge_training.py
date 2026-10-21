"""
Latent-bridge auxiliary loss: VP-diffuse a mix of two batch latents and apply the usual denoising
target via ``GaussianDiffusion.training_losses``.

Not a Schrödinger-bridge trainer; it is a regularizer that exposes the model to ``x_t`` lying
between different training examples (shuffle pairing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from diffusion.gaussian_diffusion import GaussianDiffusion


def shuffle_pair_latents(latents: torch.Tensor) -> torch.Tensor:
    """Permute the batch so (for ``B >= 2``) the permutation is never the identity."""
    b = latents.shape[0]
    if b < 2:
        return latents
    identity = torch.arange(b, device=latents.device)
    perm = torch.randperm(b, device=latents.device)
    while bool((perm == identity).all()):
        perm = torch.randperm(b, device=latents.device)
    return latents[perm]


def bridge_aux_vp_loss(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    latents: torch.Tensor,
    t: torch.Tensor,
    model_kwargs: dict,
    *,
    mix_lambda: float = 0.15,
    **loss_kwargs: Any,
) -> torch.Tensor:
    """Scalar VP loss on ``x0 = (1-λ) latents + λ shuffle_pair(latents)``.

    ``loss_kwargs`` (noise, noise_offset, loss weighting, spectral SFP) go to ``training_losses``;
    refinement and per-sample weights are always off for this auxiliary term.
    """
    lam = max(0.0, min(1.0, float(mix_lambda)))
    if lam <= 0.0:
        raise ValueError("mix_lambda must be > 0")
    x0 = (1.0 - lam) * latents + lam * shuffle_pair_latents(latents)
    return diffusion.training_losses(
        model,
        x0,
        t,
        model_kwargs=model_kwargs,
        refinement_prob=0.0,
        refinement_max_t=0,
        sample_weights=None,
        **loss_kwargs,
    )["loss"]


__all__ = ["bridge_aux_vp_loss", "shuffle_pair_latents"]
