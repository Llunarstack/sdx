"""
**CFG++** — manifold-constrained classifier-free guidance (ICLR 2025).

Lite implementation for flow/epsilon predictions: orthogonalizes the CFG update
relative to the conditional branch and uses ``cfg_lambda in [0, 1]`` (maps to
high ``omega`` in classical CFG without early-step off-manifold drift).

Chung et al. / Kim et al., arXiv:2406.08070.
"""

from __future__ import annotations

import torch


def cfg_pp_orthogonal_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Remove component of ``delta`` parallel to ``reference`` (manifold constraint)."""
    denom = (reference * reference).sum(dim=tuple(range(1, reference.ndim)), keepdim=True) + eps
    coeff = (delta * reference).sum(dim=tuple(range(1, delta.ndim)), keepdim=True) / denom
    return delta - coeff * reference


def cfg_pp_combine(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    *,
    cfg_lambda: float,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """
    CFG++ guided prediction.

    ``cfg_lambda``: typically 0.4–0.7 (similar visual strength to CFG ``omega`` 7–12).
    """
    lam = float(max(0.0, min(1.0, cfg_lambda)))
    delta = out_cond - out_uncond
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    delta_pp = cfg_pp_orthogonal_delta(delta, out_cond)
    return out_uncond + lam * delta_pp


def cfg_scale_to_pp_lambda(cfg_scale: float) -> float:
    """Heuristic map classical CFG scale to CFG++ ``lambda``."""
    w = float(max(0.0, cfg_scale))
    if w <= 1.0:
        return w
    return w / (w + 1.0)


__all__ = ["cfg_pp_combine", "cfg_pp_orthogonal_delta", "cfg_scale_to_pp_lambda"]
