"""
**QSilk Micrograin** stabilizer (CADE 2.5 appendix) — training-free latent cleanup.

Quantile clamp + light high-frequency reinjection for natural micro-texture at high res.
"""

from __future__ import annotations

import torch


def qsilk_micrograin_stabilize(
    x: torch.Tensor,
    *,
    quantile: float = 0.995,
    detail_amount: float = 0.12,
    cutoff_frac: float = 0.2,
) -> torch.Tensor:
    """
    Clamp extreme latent values, then add gated high-frequency detail residual.

    ``detail_amount=0`` → quantile clamp only.
    """
    if x.ndim != 4 or float(detail_amount) <= 0.0 and float(quantile) <= 0.0:
        return x
    q = float(max(0.9, min(0.9999, quantile)))
    flat = x.float().reshape(x.shape[0], -1)
    thr = torch.quantile(flat.abs(), q, dim=1, keepdim=True).view(-1, 1, 1, 1)
    y = x.clamp(-thr.to(dtype=x.dtype), thr.to(dtype=x.dtype))
    if float(detail_amount) <= 0.0:
        return y

    from utils.superior.frequency_cfg import _split_freq

    _low, high = _split_freq(y.float(), cutoff_frac=cutoff_frac)
    high = high.to(dtype=x.dtype)
    edge = high.abs().mean(dim=(1, 2, 3), keepdim=True)
    gate = (edge / (edge.mean() + 1e-6)).clamp(0.25, 2.0)
    return y + float(detail_amount) * gate * high


__all__ = ["qsilk_micrograin_stabilize"]
