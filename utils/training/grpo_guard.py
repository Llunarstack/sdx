"""
**GRPO-Guard** — mitigate implicit over-optimization in flow GRPO (2025).

RatioNorm on importance ratios + per-timestep gradient reweighting.

Wang et al., arXiv:2510.22319.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.training.flash_grpo import sde_discretization_lambda


@dataclass(slots=True)
class GRPOGuardConfig:
    clip_low: float = 0.8
    clip_high: float = 1.25
    ratio_norm: bool = True
    grad_reweight: bool = True


def ratio_norm_advantages(
    advantages: torch.Tensor,
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.25,
) -> torch.Tensor:
    """
    Stabilize GRPO advantages by normalizing magnitude and clipping extreme ratios.
    """
    adv = advantages.detach().float()
    if adv.numel() <= 1:
        return adv
    mu = adv.mean()
    sd = adv.std(unbiased=False) + 1e-6
    normed = (adv - mu) / sd
    imp = torch.exp(normed.clamp(-2.0, 2.0))
    imp = imp / (imp.mean() + 1e-6)
    return imp.clamp(float(clip_low), float(clip_high))


def grpo_guard_loss_weight(
    advantage: torch.Tensor,
    t_frac: float | torch.Tensor,
    *,
    config: GRPOGuardConfig | None = None,
) -> torch.Tensor:
    """Per-sample loss weight with optional timestep reweighting."""
    cfg = config or GRPOGuardConfig()
    w = ratio_norm_advantages(advantage.view(-1), clip_low=cfg.clip_low, clip_high=cfg.clip_high)
    w = torch.exp(-w.clamp(0.5, 2.0))
    if cfg.grad_reweight:
        lam = sde_discretization_lambda(t_frac)
        if isinstance(lam, torch.Tensor):
            w = w / lam.clamp_min(1e-4)
        else:
            w = w / max(float(lam), 1e-4)
    return w


def grpo_guard_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantages: torch.Tensor,
    t_fracs: torch.Tensor,
    *,
    config: GRPOGuardConfig | None = None,
    clip: float = 3.0,
) -> torch.Tensor:
    """Combine GRPO weighting with GRPO-Guard ratio/timestep correction."""
    cfg = config or GRPOGuardConfig()
    total = per_sample_loss.new_zeros(())
    wsum = per_sample_loss.new_zeros(())
    for i in range(per_sample_loss.shape[0]):
        adv_i = advantages[i : i + 1] if advantages.ndim else advantages[i]
        tf = t_fracs[i] if t_fracs.numel() > i else t_fracs.reshape(-1)[0]
        gw = grpo_guard_loss_weight(adv_i, tf, config=cfg)
        total = total + per_sample_loss[i] * gw.squeeze()
        wsum = wsum + gw.squeeze()
    return total / (wsum + 1e-8)


__all__ = [
    "GRPOGuardConfig",
    "grpo_guard_loss_weight",
    "grpo_guard_weighted_loss",
    "ratio_norm_advantages",
]
