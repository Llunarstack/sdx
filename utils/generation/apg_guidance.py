"""
**Adaptive Projected Guidance (APG)** for classifier-free guidance.

Decomposes the CFG delta into components parallel vs orthogonal to the conditional
prediction, then down-weights the parallel part (reduces oversaturation at high CFG).

Sadat et al., ICLR 2025 — "Eliminating Oversaturation and Artifacts of High Guidance
Scales in Diffusion Models". Works on flow velocities or noise predictions alike.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def decompose_parallel_orthogonal(
    delta: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ``delta_parallel = proj_ref(delta)``, ``delta_orth = delta - delta_parallel``.

    Reference is typically the conditional model output (velocity / x0 / noise).
    """
    ref = reference
    denom = (ref * ref).sum(dim=tuple(range(1, ref.ndim)), keepdim=True) + eps
    coeff = (delta * ref).sum(dim=tuple(range(1, delta.ndim)), keepdim=True) / denom
    parallel = coeff * ref
    orth = delta - parallel
    return parallel, orth


def apg_guidance_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    *,
    parallel_eta: float = 0.0,
    cfg_rescale: float = 0.0,
    momentum_delta: Optional[torch.Tensor] = None,
    momentum_beta: float = 0.2,
) -> torch.Tensor:
    """
    APG-modified CFG delta.

    ``parallel_eta``: weight on parallel component (0 = remove parallel, 1 = standard CFG).
    Optional reverse momentum blends previous step delta.
    """
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    parallel, orth = decompose_parallel_orthogonal(delta, reference)
    eta = float(max(0.0, min(1.0, parallel_eta)))
    out = eta * parallel + orth
    if momentum_delta is not None and float(momentum_beta) > 0.0:
        b = float(momentum_beta)
        out = (1.0 - b) * out + b * momentum_delta
    return out


def apg_cfg_combine(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    parallel_eta: float = 0.0,
    cfg_rescale: float = 0.0,
    momentum_delta: Optional[torch.Tensor] = None,
    momentum_beta: float = 0.2,
) -> torch.Tensor:
    """``out_uncond + cfg_scale * APG(delta)`` with ``delta = out_cond - out_uncond``."""
    delta = out_cond - out_uncond
    apg_delta = apg_guidance_delta(
        delta,
        out_cond,
        parallel_eta=parallel_eta,
        cfg_rescale=cfg_rescale,
        momentum_delta=momentum_delta,
        momentum_beta=momentum_beta,
    )
    return out_uncond + float(cfg_scale) * apg_delta


__all__ = [
    "apg_cfg_combine",
    "apg_guidance_delta",
    "decompose_parallel_orthogonal",
]
