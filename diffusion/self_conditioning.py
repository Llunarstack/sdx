"""
Self-conditioning helpers for diffusion models.

Why self-conditioning
---------------------
Normally each denoising step predicts the clean sample (x0/epsilon) from scratch.
Self-conditioning (Chen et al., "Analog Bits") lets the model *also* see its own
previous estimate as an extra input, so successive steps stay consistent and
refine rather than restart. It's a cheap, well-established quality boost.

Pattern:
1) Predict a provisional x0 / epsilon.
2) Feed a **detached** projection of that prediction back into the model on a
   second pass. Detaching is essential: the self-cond input is conditioning, not
   something we backprop through, so gradients must not flow into the first pass.
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
    Return the detached prediction to use as a self-conditioning signal.

    During training we randomly drop the self-cond signal (``drop_prob``) so the
    model also learns to denoise *without* it — otherwise inference, where the
    first step has no prior estimate, would be out of distribution.
    """
    if not enabled or prediction is None:
        return None
    # Coin-flip dropout: skip self-cond on this step so the model stays robust
    # to its absence (notably the very first sampling step).
    if training and float(drop_prob) > 0.0:
        if torch.rand(1, device=prediction.device).item() < float(drop_prob):
            return None
    # Detach so this acts purely as conditioning, never as a gradient path.
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
