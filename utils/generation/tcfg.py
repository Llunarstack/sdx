"""
**TCFG** — Tangential Damping Classifier-free Guidance (CVPR 2025).

Lite training-free variant: damp components of the unconditional prediction that are
tangential (orthogonal) to the conditional direction, keeping the denoise trajectory
closer to the conditional manifold.

Kwon et al., arXiv:2412.12095.
"""

from __future__ import annotations

import torch

from utils.generation.cfg_pp import cfg_pp_orthogonal_delta


def tcfg_damp_unconditional(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    damping: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return a damped unconditional prediction with tangential components reduced.

    ``damping=1`` removes tangential drift fully; ``0`` leaves ``out_uncond`` unchanged.
    """
    d = float(max(0.0, min(1.0, damping)))
    if d <= 0.0:
        return out_uncond
    delta_u = out_uncond - out_cond
    tang = cfg_pp_orthogonal_delta(delta_u, out_cond, eps=eps)
    return out_uncond - d * tang


def tcfg_combine(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    damping: float = 1.0,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """Standard CFG with tangentially damped unconditional branch."""
    u_adj = tcfg_damp_unconditional(out_cond, out_uncond, damping=float(damping))
    delta = out_cond - u_adj
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    return u_adj + float(cfg_scale) * delta


__all__ = ["tcfg_combine", "tcfg_damp_unconditional"]
