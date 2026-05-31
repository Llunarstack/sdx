"""
**Rectified-CFG++** for flow-matching sampling.

Refines CFG delta using signal-norm rescale + optional tangent normalization
on the velocity field (reduces overshoot at high CFG on flow models).
"""

from __future__ import annotations

import torch


def rectified_cfgpp_combine(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    *,
    cfg_scale: float,
    cfg_rescale: float = 0.0,
    tangent_norm: float = 0.0,
) -> torch.Tensor:
    """
    CFG on flow velocity predictions with Rectified-CFG++ style stabilization.

    ``tangent_norm > 0``: clamp delta norm relative to cond norm (reduces halos).
    """
    delta = v_cond - v_uncond
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    if float(tangent_norm) > 0.0:
        cn = v_cond.norm() + 1e-8
        dn = delta.norm()
        cap = float(tangent_norm) * cn
        if dn > cap:
            delta = delta * (cap / dn)
    return v_uncond + float(cfg_scale) * delta


__all__ = ["rectified_cfgpp_combine"]
