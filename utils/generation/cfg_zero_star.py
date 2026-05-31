"""
**CFG-Zero★** — improved classifier-free guidance for flow-matching models.

- **Optimized scale** ``st*``: corrects velocity mismatch between cond/uncond branches.
- **Zero-init**: zero velocity on the first ~4% of ODE steps when the model is underfitted.

Fan et al., arXiv:2503.18886 (2025).
"""

from __future__ import annotations

import torch


def cfg_zero_optimized_scale(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ``st* = <v_cond, v_uncond> / ||v_uncond||^2`` per batch element.

    Returns broadcastable tensor matching ``v_cond`` layout.
    """
    dims = tuple(range(1, v_cond.ndim))
    dot = (v_cond * v_uncond).sum(dim=dims, keepdim=True)
    un_sq = (v_uncond * v_uncond).sum(dim=dims, keepdim=True) + eps
    st = dot / un_sq
    return st.to(dtype=v_cond.dtype)


def cfg_zero_star_combine(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    sample_step: int = 0,
    total_steps: int = 1,
    zero_init_frac: float = 0.04,
    use_optimized_scale: bool = True,
) -> torch.Tensor:
    """
    CFG-Zero★ guided velocity.

    Early steps (``sample_step < zero_init_frac * total_steps``): return zeros.
    Otherwise: ``v_u * st* + w * (v_c - v_u * st*)``.
    """
    tot = max(1, int(total_steps))
    z_steps = max(1, int(round(float(zero_init_frac) * tot)))
    if int(sample_step) < z_steps:
        return torch.zeros_like(v_cond)

    if not use_optimized_scale:
        return v_uncond + float(cfg_scale) * (v_cond - v_uncond)

    st = cfg_zero_optimized_scale(v_cond, v_uncond)
    return v_uncond * st + float(cfg_scale) * (v_cond - v_uncond * st)


def zero_init_step_count(total_steps: int, *, zero_init_frac: float = 0.04) -> int:
    """Number of initial ODE steps to zero out."""
    return max(1, int(round(float(zero_init_frac) * max(1, int(total_steps)))))


__all__ = [
    "cfg_zero_optimized_scale",
    "cfg_zero_star_combine",
    "zero_init_step_count",
]
