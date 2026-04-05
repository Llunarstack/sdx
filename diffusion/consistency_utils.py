"""
Consistency-style utilities for diffusion training/inference experiments.

These are lightweight tensor helpers (no architecture assumptions) meant to
support consistency/distillation-style objectives and refinement loops.
"""

from __future__ import annotations

import torch


def temporal_ema_target(
    prev_target: torch.Tensor | None,
    current: torch.Tensor,
    *,
    decay: float = 0.95,
) -> torch.Tensor:
    """
    Exponential moving average target:
      y_t = d * y_{t-1} + (1-d) * current
    """
    d = max(0.0, min(0.9999, float(decay)))
    if prev_target is None:
        return current.detach()
    if prev_target.shape != current.shape:
        raise ValueError("prev_target/current shapes must match")
    return (d * prev_target + (1.0 - d) * current.detach()).detach()


def consistency_delta_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L2 agreement loss between two predictions from nearby timesteps/noise states.
    """
    if pred_a.shape != pred_b.shape:
        raise ValueError("pred_a/pred_b shapes must match")
    d = (pred_a - pred_b).pow(2)
    if reduction == "none":
        return d
    if reduction == "sum":
        return d.sum()
    if reduction != "mean":
        raise ValueError("reduction must be one of: mean | sum | none")
    return d.mean()


def one_step_consistency_refine(
    x0_pred: torch.Tensor,
    x0_teacher: torch.Tensor,
    *,
    step_size: float = 0.15,
) -> torch.Tensor:
    """
    Move x0_pred one small step toward teacher target.
    """
    a = max(0.0, min(1.0, float(step_size)))
    if x0_pred.shape != x0_teacher.shape:
        raise ValueError("x0_pred/x0_teacher shapes must match")
    return x0_pred + a * (x0_teacher - x0_pred)

