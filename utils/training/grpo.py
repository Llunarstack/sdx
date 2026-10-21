"""**BranchGRPO** scaffold — tree-structured rollouts for diffusion GRPO (2025).

At scheduled split steps, trajectories branch into multiple children while reusing shared
prefix computation. Depth-wise reward fusion aggregates leaf scores back to branches.

Inspired by BranchGRPO (structured branching in diffusion models).

Also holds what used to be: branch_grpo, dense_grpo, flash_grpo, grpo_guard, turning_point_grpo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from utils.training.flow_grpo import decode_latent_to_rgb_uint8, group_relative_advantages

# ==================================================================================================
# branch_grpo: **BranchGRPO** scaffold — tree-structured rollouts for diffusion GRPO (2025).
#
# At scheduled split steps, trajectories branch into multiple children while reusing shared
# prefix computation. Depth-wise reward fusion aggregates leaf scores back to branches.
#
# Inspired by BranchGRPO (structured branching in diffusion models).
# ==================================================================================================


@dataclass(slots=True)
class BranchGRPOConfig:
    branch_factor: int = 2
    """Children per split node."""
    split_step_fractions: tuple[float, ...] = (0.35, 0.65)
    """Fractions along denoise trajectory where branching occurs."""
    fuse: str = "mean"
    """Leaf reward fusion: mean | max | min."""


def split_steps_from_fractions(total_steps: int, fractions: Sequence[float]) -> list[int]:
    """Map fractional positions to integer step indices in ``[0, total_steps)``."""
    n = max(1, int(total_steps))
    out: list[int] = []
    for f in fractions:
        idx = int(round(float(f) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        if not out or out[-1] != idx:
            out.append(idx)
    return out


def enumerate_branch_paths(
    total_steps: int,
    *,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
) -> list[tuple[int, ...]]:
    """
    Return leaf paths as tuples of branch choices (0..branch_factor-1) per split.

    Does not run the model — planning helper for trainers.
    """
    splits = split_steps_from_fractions(total_steps, split_fractions)
    if not splits:
        return [(0,)]
    k = max(1, int(branch_factor))
    paths: list[tuple[int, ...]] = [(0,)]
    for _ in splits:
        paths = [p + (b,) for p in paths for b in range(k)]
    return paths


def branch_relative_advantages(
    path_rewards: torch.Tensor,
    path_ids: Sequence[str],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    GRPO-style advantages over branch leaves sharing the same prompt.

    ``path_rewards``: (P,), one reward per leaf path.
    """
    r = path_rewards.detach().float().view(-1)
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sd = r.std(unbiased=False)
    return ((r - mu) / (sd + eps)).clamp(-3.0, 3.0)


def branch_rollout_flow_samples(
    rollout_fn,
    *,
    num_paths: int,
    steps: int,
    branch_factor: int = 2,
    split_fractions: Sequence[float] = (0.35, 0.65),
    base_seed: int = 0,
) -> list[torch.Tensor]:
    """
    Run multiple rollout paths with distinct seeds (BranchGRPO scaffold).

    Full prefix-sharing would require hooking inside ``sample_loop``; this helper
    reuses the planning API and runs independent short rollouts per leaf path.
    """
    paths = enumerate_branch_paths(
        int(steps),
        branch_factor=int(branch_factor),
        split_fractions=split_fractions,
    )
    n = max(1, int(num_paths))
    paths = paths[:n] if len(paths) >= n else paths + [paths[-1]] * (n - len(paths))
    latents: list[torch.Tensor] = []
    for i, _path in enumerate(paths):
        if base_seed >= 0:
            torch.manual_seed(int(base_seed) + i)
        latents.append(rollout_fn())
    return latents


# ==================================================================================================
# dense_grpo: **DenseGRPO** helpers — step-wise dense rewards for flow-matching alignment (2026).
#
# Estimates per-denoise-step reward gain via short ODE decode of intermediate latents,
# plus reward-aware SDE noise scaling (calibrated exploration by timestep).
# ==================================================================================================


@dataclass(slots=True)
class DenseGRPOConfig:
    ode_refine_steps: int = 4
    """Short deterministic ODE steps to estimate x0 from x_t."""
    reward_aware_sde: bool = True
    base_sde_scale: float = 0.35


def flow_fraction_from_t_index(t_idx: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Map discrete VP index to flow fraction in [0, 1] (noise → data)."""
    T = max(1, int(num_timesteps) - 1)
    return t_idx.detach().float().clamp(0, T) / float(T)


def estimate_x0_from_xt_flow(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    t_frac: torch.Tensor,
) -> torch.Tensor:
    """One-step x0 estimate from flow velocity: ``x0 ≈ x_t - t * v``."""
    tv = t_frac.view(-1, 1, 1, 1).to(device=x_t.device, dtype=x_t.dtype)
    return x_t - tv * v_pred


def short_ode_refine_x0(
    model: torch.nn.Module,
    diffusion: object,
    x_t: torch.Tensor,
    t_start: torch.Tensor,
    *,
    model_kwargs: dict,
    steps: int = 4,
    flow: bool = True,
) -> torch.Tensor:
    """
    Deterministic short rollout from ``x_t`` toward x0 (DenseGRPO ODE decode) with ``sample_loop``'s ``x_init``
    resume; ``flow`` selects the flow-matching or VP trajectory to match the checkpoint.
    """
    if not hasattr(diffusion, "sample_loop"):
        v = model(x_t, t_start, **model_kwargs)
        if v.shape[1] > x_t.shape[1]:
            v = v[:, : x_t.shape[1]]
        tf = flow_fraction_from_t_index(t_start, int(getattr(diffusion, "num_timesteps", 1000)))
        return estimate_x0_from_xt_flow(x_t, v, tf)
    t0 = int(t_start.reshape(-1)[0].detach().cpu().item())
    return diffusion.sample_loop(
        model,
        x_t.shape,
        model_kwargs_cond=model_kwargs,
        model_kwargs_uncond=None,
        cfg_scale=1.0,
        num_inference_steps=max(1, int(steps)),
        device=str(x_t.device),
        dtype=x_t.dtype,
        x_init=x_t,
        start_timestep=t0,
        flow_matching_sample=bool(flow),
        flow_solver="euler",
    )


def step_advantages_from_gains(gains: Sequence[float], *, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages on dense step gains (one timestep, G samples)."""
    return group_relative_advantages(torch.tensor(list(gains), dtype=torch.float32), eps=eps)


def score_latent_reward(
    x0_hat: torch.Tensor,
    prompt: str,
    *,
    vae: torch.nn.Module,
    score_fn: Callable[[np.ndarray, str], float],
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    rae_bridge: object = None,
) -> float:
    """Decode latent estimate and run reward model."""
    rgb = decode_latent_to_rgb_uint8(x0_hat, vae, latent_scale=latent_scale, ae_type=ae_type, rae_bridge=rae_bridge)
    return float(score_fn(rgb, prompt))


# ==================================================================================================
# flash_grpo: **Flash-GRPO** helpers — efficient single-step / iso-temporal GRPO (2025–2026).
#
# - **Iso-temporal grouping**: all rollouts in a GRPO group share the same denoise timestep
#   for SDE exploration so advantages reflect policy quality, not timestep difficulty.
# - **Temporal gradient rectification**: normalize policy gradients by SDE discretization scale λ(t).
#
# Inspired by Flash-GRPO (video alignment); applicable to image DiT flow training scaffolds.
# ==================================================================================================


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


# ==================================================================================================
# grpo_guard: **GRPO-Guard** — mitigate implicit over-optimization in flow GRPO (2025).
#
# RatioNorm on importance ratios + per-timestep gradient reweighting.
#
# Wang et al., arXiv:2510.22319.
# ==================================================================================================


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


def grpo_guard_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantages: torch.Tensor,
    t_fracs: torch.Tensor,
    *,
    config: GRPOGuardConfig | None = None,
) -> torch.Tensor:
    """Reward-weighted loss over one rollout group with GRPO-Guard's corrections: RatioNorm-normalised weights
    (increasing in the advantage, clipped to ``[clip_low, clip_high]``) and ``1 / lambda(t)`` per-timestep gradient
    reweighting. Weights are normalised over the group, so pass all of its samples at once."""
    cfg = config or GRPOGuardConfig()
    adv = advantages.detach().float().view(-1)
    if adv.numel() < 2:
        raise ValueError("grpo_guard_weighted_loss needs the whole group (>= 2 samples)")
    if cfg.ratio_norm:
        w = ratio_norm_advantages(adv, clip_low=cfg.clip_low, clip_high=cfg.clip_high)
    else:
        w = torch.exp(adv.clamp(-3.0, 3.0))
    if cfg.grad_reweight:
        w = w / sde_discretization_lambda(t_fracs.detach().float().view(-1).expand_as(w)).clamp_min(1e-4)
    w = w.to(per_sample_loss.dtype)
    return (per_sample_loss.view(-1) * w).sum() / (w.sum() + 1e-8)


# ==================================================================================================
# turning_point_grpo: **TurningPoint-GRPO (TP-GRPO)** — step incremental rewards + turning-point long-term credit.
#
# Alleviates sparse terminal rewards in flow-based GRPO by:
# 1. Incremental step rewards ``ΔR_t = R_t - R_{t-1}``
# 2. Turning points where ``sign(ΔR_t) != sign(ΔR_{t+1})`` get aggregated long-term bonus
#
# Tong et al., arXiv:2602.06422 (2026).
# ==================================================================================================


@dataclass(slots=True)
class TurningPointGRPOConfig:
    long_term_weight: float = 0.5
    """Weight on aggregated reward assigned at turning-point steps."""
    use_incremental_only: bool = False
    """If True, skip turning-point bonus (ablation)."""


def incremental_rewards_from_trajectory(step_rewards: Sequence[float]) -> list[float]:
    """``ΔR_t`` from absolute per-step rewards ``R_t``."""
    prev = 0.0
    out: list[float] = []
    for r in step_rewards:
        rv = float(r)
        out.append(rv - prev)
        prev = rv
    return out


def detect_turning_point_indices(incremental: Sequence[float]) -> list[int]:
    """
    Turning point at step ``t`` when ``sign(ΔR_t) != sign(ΔR_{t+1})`` (non-zero signs).
    """
    inc = [float(x) for x in incremental]
    if len(inc) < 2:
        return []
    tps: list[int] = []
    for t in range(len(inc) - 1):
        a, b = inc[t], inc[t + 1]
        if a == 0.0 or b == 0.0:
            continue
        if (a > 0) != (b > 0):
            tps.append(t)
    return tps


def long_term_aggregate_reward(
    incremental: Sequence[float],
    terminal_reward: float,
) -> float:
    """Simple long-term signal: terminal minus mean incremental."""
    inc = list(incremental)
    if not inc:
        return float(terminal_reward)
    return float(terminal_reward) - sum(inc) / len(inc)


def tp_grpo_step_weights(
    step_rewards: Sequence[float],
    terminal_reward: float,
    *,
    config: TurningPointGRPOConfig | None = None,
) -> list[float]:
    """
    Per-step training weights combining incremental + turning-point long-term bonus.

    Returns one scalar weight per denoise step (same length as ``step_rewards``).
    """
    cfg = config or TurningPointGRPOConfig()
    inc = incremental_rewards_from_trajectory(step_rewards)
    if not inc:
        return [float(terminal_reward)]

    weights = list(inc)
    if not cfg.use_incremental_only:
        lt = long_term_aggregate_reward(inc, terminal_reward)
        for t in detect_turning_point_indices(inc):
            weights[t] = weights[t] + float(cfg.long_term_weight) * lt
    return weights


__all__ = [
    "BranchGRPOConfig",
    "branch_relative_advantages",
    "branch_rollout_flow_samples",
    "enumerate_branch_paths",
    "split_steps_from_fractions",
    "DenseGRPOConfig",
    "estimate_x0_from_xt_flow",
    "flow_fraction_from_t_index",
    "score_latent_reward",
    "short_ode_refine_x0",
    "step_advantages_from_gains",
    "iso_temporal_group_advantages",
    "rectify_policy_gradient",
    "sample_iso_temporal_index",
    "sde_discretization_lambda",
    "GRPOGuardConfig",
    "grpo_guard_weighted_loss",
    "ratio_norm_advantages",
    "TurningPointGRPOConfig",
    "detect_turning_point_indices",
    "incremental_rewards_from_trajectory",
    "long_term_aggregate_reward",
    "tp_grpo_step_weights",
]
