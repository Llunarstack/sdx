"""
**Flash-GRPO** helpers — efficient single-step / iso-temporal GRPO (2025–2026).

- **Iso-temporal grouping**: all rollouts in a GRPO group share the same denoise timestep
  for SDE exploration so advantages reflect policy quality, not timestep difficulty.
- **Temporal gradient rectification**: normalize policy gradients by SDE discretization scale λ(t).

Inspired by Flash-GRPO (video alignment); applicable to image DiT flow training scaffolds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FlashGRPOConfig:
    """Knobs for Flash-GRPO style training."""

    iso_temporal: bool = True
    rectified_gradients: bool = True
    sde_lambda_floor: float = 1e-4


def sample_iso_temporal_index(
    num_timesteps: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one timestep index in ``[0, num_timesteps)`` for a whole GRPO group."""
    T = max(1, int(num_timesteps))
    return torch.randint(0, T, (1,), device=device, generator=generator, dtype=torch.long)


def sde_discretization_lambda(t_frac: float | torch.Tensor, *, floor: float = 1e-4) -> float | torch.Tensor:
    """
    Time-dependent SDE scaling λ(t) ∝ sqrt(t(1-t)) on flow fraction.

    Used to rectify gradient magnitude across timesteps (Flash-GRPO).
    """
    if isinstance(t_frac, torch.Tensor):
        u = t_frac.detach().float().clamp(0.0, 1.0)
        lam = torch.sqrt(u * (1.0 - u) + float(floor))
        return lam
    u = float(max(0.0, min(1.0, float(t_frac))))
    return float(max((u * (1.0 - u)) ** 0.5, floor**0.5))


def rectify_policy_gradient(
    grad_or_loss_scale: torch.Tensor,
    t_frac: float | torch.Tensor,
    *,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Divide by λ(t) so mid-trajectory steps do not dominate updates."""
    lam = sde_discretization_lambda(t_frac, floor=floor)
    if isinstance(lam, torch.Tensor):
        return grad_or_loss_scale / lam.clamp_min(float(floor))
    return grad_or_loss_scale / max(float(lam), float(floor))


def iso_temporal_group_advantages(
    rewards: torch.Tensor,
    timestep_indices: torch.Tensor,
    *,
    num_timesteps: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRPO advantages normalized **within (timestep bucket, prompt group)**.

    ``rewards``: (N,), ``timestep_indices``: (N,) discrete step indices.
    """
    r = rewards.detach().float().view(-1)
    t = timestep_indices.detach().long().view(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    adv = torch.zeros_like(r)
    for ti in t.unique().tolist():
        mask = t == int(ti)
        if mask.sum() <= 1:
            continue
        g = r[mask]
        mu = g.mean()
        sd = g.std(unbiased=False)
        adv[mask] = (g - mu) / (sd + eps)
    if adv.abs().sum() == 0:
        mu = r.mean()
        sd = r.std(unbiased=False)
        adv = (r - mu) / (sd + eps)
    return adv.clamp(-3.0, 3.0)


def flash_grpo_loss_weight(
    advantage: torch.Tensor,
    t_frac: float | torch.Tensor,
    *,
    rectified: bool = True,
    clip: float = 3.0,
) -> torch.Tensor:
    """Per-sample GRPO weight with optional temporal rectification."""
    adv = advantage.detach().float().clamp(-float(clip), float(clip))
    w = torch.exp(-adv / max(float(clip), 1e-6))
    if rectified:
        lam = sde_discretization_lambda(t_frac)
        if isinstance(lam, torch.Tensor):
            w = w / lam.clamp_min(1e-4)
        else:
            w = w / max(float(lam), 1e-4)
    return w


__all__ = [
    "FlashGRPOConfig",
    "flash_grpo_loss_weight",
    "iso_temporal_group_advantages",
    "rectify_policy_gradient",
    "sample_iso_temporal_index",
    "sde_discretization_lambda",
]
