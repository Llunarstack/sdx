"""
Self-conditioning helpers for diffusion models.

Pattern:
1) Predict a provisional x0 / epsilon.
2) Feed a detached projection of this prediction back to model in second pass.
"""

from __future__ import annotations

from typing import Optional

import torch


def maybe_detached_self_cond(
    prediction: Optional[torch.Tensor],
    *,
    enabled: bool,
    drop_prob: float = 0.5,
    training: bool = True,
) -> Optional[torch.Tensor]:
    """
    Return detached prediction as self-conditioning signal.
    Optionally dropout self-cond during training for robustness.
    """
    if not enabled or prediction is None:
        return None
    if training and float(drop_prob) > 0.0:
        if torch.rand(1, device=prediction.device).item() < float(drop_prob):
            return None
    return prediction.detach()


def blend_self_cond(
    x: torch.Tensor,
    self_cond: Optional[torch.Tensor],
    *,
    strength: float = 0.5,
) -> torch.Tensor:
    """
    Blend tensor with self-conditioning tensor:
      out = (1-s) * x + s * self_cond
    """
    if self_cond is None:
        return x
    s = max(0.0, min(1.0, float(strength)))
    if self_cond.shape != x.shape:
        raise ValueError(f"self_cond shape {tuple(self_cond.shape)} must match x shape {tuple(x.shape)}")
    return (1.0 - s) * x + s * self_cond.to(dtype=x.dtype)

