"""
**DenseGRPO** helpers — step-wise dense rewards for flow-matching alignment (2026).

Estimates per-denoise-step reward gain via short ODE decode of intermediate latents,
plus reward-aware SDE noise scaling (calibrated exploration by timestep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import torch

from utils.training.flow_grpo import decode_latent_to_rgb_uint8, group_relative_advantages


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


def reward_aware_sde_scale(t_frac: float | torch.Tensor, *, base: float = 0.35) -> float | torch.Tensor:
    """
    Calibrate SDE exploration: peak noise injection mid-trajectory (t≈0.5).

    DenseGRPO finds uniform SDE scale mismatches time-varying flow noise intensity.
    """
    if isinstance(t_frac, torch.Tensor):
        u = t_frac.detach().float().clamp(0.0, 1.0)
        return float(base) * (4.0 * u * (1.0 - u))
    u = float(max(0.0, min(1.0, float(t_frac))))
    return float(base) * (4.0 * u * (1.0 - u))


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
) -> torch.Tensor:
    """
    Deterministic short rollout from ``x_t`` toward x0 (DenseGRPO ODE decode).

    Uses flow-matching sample loop with ``x_init`` and reduced steps.
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
        flow_matching_sample=True,
        flow_solver="euler",
    )


def dense_reward_gains(
    terminal_rewards: Sequence[float],
    step_rewards: Sequence[Sequence[float]],
) -> List[List[float]]:
    """
    Convert per-step absolute rewards into incremental gains ``ΔR_t``.

    ``step_rewards[g][t]`` is reward at step t for group sample g;
    returns ``ΔR_t = R_t - R_{t-1}`` with ``R_{-1} = 0``.
    """
    gains: List[List[float]] = []
    for g, traj in enumerate(step_rewards):
        prev = 0.0
        row: List[float] = []
        for r in traj:
            rv = float(r)
            row.append(rv - prev)
            prev = rv
        if not row and g < len(terminal_rewards):
            row = [float(terminal_rewards[g])]
        gains.append(row)
    return gains


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


__all__ = [
    "DenseGRPOConfig",
    "dense_reward_gains",
    "estimate_x0_from_xt_flow",
    "flow_fraction_from_t_index",
    "reward_aware_sde_scale",
    "score_latent_reward",
    "short_ode_refine_x0",
    "step_advantages_from_gains",
]
