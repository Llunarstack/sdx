"""
**Flow-GRPO** scaffold for online RL on flow-matching / VP DiT models.

Inspired by Flow-GRPO (NeurIPS 2025): ODE→SDE noise injection for exploration,
terminal reward from ``CompositeRanker('online').score_one``, group-relative policy update on
denoising surrogate loss.

This is a **practical SDX trainer** — not a full reproduction of multi-GPU GRPO,
but a working loop: sample → score → weighted fine-tune on high-reward trajectories.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(slots=True)
class FlowGRPOConfig:
    num_samples: int = 4
    train_steps_per_prompt: int = 1
    sde_noise_scale: float = 0.35
    """Injected noise for SDE-style exploration during rollouts."""
    kl_coef: float = 0.02
    """MSE penalty vs reference (anti reward hacking)."""
    advantage_clip: float = 3.0
    reward_temperature: float = 1.0
    denoise_steps: int = 8
    """Reduced steps during rollout (Flow-GRPO denoising reduction)."""


def flow_sde_step(
    x: torch.Tensor,
    v_pred: torch.Tensor,
    s_cur: float,
    s_next: float,
    *,
    noise_level: float = 0.7,
) -> torch.Tensor:
    """
    One step of Flow-GRPO's marginal-preserving reverse SDE (Liu et al. 2025, Eq. 9) for ``x_s = (1-s) x0 + s eps``:

    ``x' = x - [v + sigma^2/(2s) (x + (1-s) v)] ds + sigma sqrt(ds) z``, ``sigma = noise_level * sqrt(s / (1-s))``,
    ``ds = s_cur - s_next > 0``. The drift adds ``sigma^2/2`` times the score ``-(x + (1-s) v)/s`` so every ``x_s`` keeps
    the probability-flow ODE's marginal while the injected noise gives GRPO distinct rollouts to compare.
    """
    ds = float(s_cur) - float(s_next)
    # sigma is unbounded at s = 1; like the reference code, use the next grid point in the denominator there.
    denom = 1.0 - (float(s_next) if float(s_cur) >= 1.0 else float(s_cur))
    sigma = float(noise_level) * (float(s_cur) / max(denom, 1e-6)) ** 0.5
    drift = v_pred + (sigma**2 / (2.0 * max(float(s_cur), 1e-6))) * (x + (1.0 - float(s_cur)) * v_pred)
    return x - drift * ds + sigma * ds**0.5 * torch.randn_like(x)


def group_relative_advantages(rewards: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """GRPO-style advantages: (r - mean) / (std + eps) within group."""
    r = rewards.detach().float()
    if r.numel() <= 1:
        return torch.zeros_like(r)
    mu = r.mean()
    sd = r.std(unbiased=False)
    return ((r - mu) / (sd + eps)).clamp(-3.0, 3.0)


def grpo_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float = 3.0,
    temperature: float = 1.0,
    extra_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reward-weighted regression over one rollout group (Peters & Schaal 2007; Lee et al. 2023 for text-to-image):
    ``sum_i w_i L_i / sum_i w_i`` with ``w_i = exp(clip(A_i) / temperature)``, so the denoising loss is fitted hardest
    on the highest-reward rollouts. ``extra_weights`` (e.g. per-timestep factors) multiply ``w``.

    The weights are normalised over the group, so pass all of its samples at once: a single sample's weight cancels.
    """
    if per_sample_loss.numel() < 2:
        raise ValueError("grpo_weighted_loss needs the whole group (>= 2 samples); one sample's weight cancels out")
    adv = advantages.detach().float().view(-1).clamp(-float(clip), float(clip))
    w = torch.exp(adv / max(float(temperature), 1e-6))
    if extra_weights is not None:
        w = w * extra_weights.detach().float().view(-1)
    w = w.to(per_sample_loss.dtype)
    return (per_sample_loss.view(-1) * w).sum() / (w.sum() + 1e-8)


def reference_kl_penalty(
    policy_pred: torch.Tensor,
    ref_pred: torch.Tensor,
    *,
    coef: float = 0.02,
) -> torch.Tensor:
    """Simple MSE KL surrogate between policy and frozen reference."""
    if coef <= 0.0:
        return policy_pred.new_zeros(())
    return float(coef) * F.mse_loss(policy_pred, ref_pred.detach())


def decode_latent_to_rgb_uint8(
    latent: torch.Tensor,
    vae: torch.nn.Module,
    *,
    latent_scale: float = 0.18215,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> np.ndarray:
    """Decode single latent (1,C,H,W) to RGB uint8 numpy."""
    z = latent / float(latent_scale) if str(ae_type) == "kl" else latent
    if str(ae_type) == "rae" and rae_bridge is not None:
        z = rae_bridge.dit_to_rae(z)
    decoded = vae.decode(z)
    img01 = ((decoded.sample if hasattr(decoded, "sample") else decoded) * 0.5 + 0.5).clamp(0, 1)
    return (img01[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def rollout_flow_sample(
    model: torch.nn.Module,
    diffusion: Any,
    shape: tuple[int, ...],
    *,
    model_kwargs_cond: dict[str, Any],
    model_kwargs_uncond: dict[str, Any] | None,
    cfg_scale: float,
    steps: int,
    device: str,
    sde_noise_scale: float = 0.35,
    flow: bool = True,
    flow_shift: float = 1.0,
) -> torch.Tensor:
    """
    Exploratory rollout for GRPO. Flow models integrate :func:`flow_sde_step` on the (SD3-shifted) time grid with CFG;
    VP models use ``sample_loop`` with DDIM ``eta = sde_noise_scale`` (eta = 1 is ancestral DDPM).
    """
    if not flow:
        return diffusion.sample_loop(
            model,
            shape,
            model_kwargs_cond=model_kwargs_cond,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=float(cfg_scale),
            num_inference_steps=max(2, int(steps)),
            eta=float(sde_noise_scale),
            timestep_schedule="euler",
            device=device,
            dtype=torch.float32,
        )
    from diffusion.flow_matching import flow_time_to_index
    from diffusion.schedules import time_shift

    num_t = int(diffusion.num_timesteps)
    grid = time_shift(torch.linspace(1.0, 0.0, max(2, int(steps)) + 1), float(flow_shift)).tolist()
    x = torch.randn(shape, device=device, dtype=torch.float32)
    guided = model_kwargs_uncond is not None and float(cfg_scale) != 1.0
    for s_cur, s_next in zip(grid[:-1], grid[1:]):
        t = flow_time_to_index(torch.full((shape[0],), s_cur, device=device), num_t)
        v = model(x, t, **model_kwargs_cond)[:, : shape[1]].float()
        if guided:
            v_u = model(x, t, **model_kwargs_uncond)[:, : shape[1]].float()
            v = v_u + float(cfg_scale) * (v - v_u)
        x = flow_sde_step(x, v, s_cur, s_next, noise_level=sde_noise_scale) if s_next > 0 else x - s_cur * v
    return x


@dataclass(slots=True)
class SRPOConfig:
    """SRPO realism controls (Tencent Hunyuan 2025, "Directly Aligning the Full Diffusion Trajectory")."""

    control_positive: str = "natural skin texture, realistic photograph, true color"
    control_negative: str = "oily glossy skin, waxy plastic render, ai generated look"
    late_timestep_cutoff: float = 0.75
    """Trajectory fraction (0=noise, 1=clean) past which updates are masked: late steps over-polish skin."""
    reward_temperature: float = 1.0


def semantic_relative_reward(
    score_fn: Callable[[Any, str], Any], images: Any, prompt: str, *, config: SRPOConfig
) -> Any:
    """``score(prompt + positive) - score(prompt + negative)``: the reward model's global aesthetic bias
    (the shared "AI look") cancels and only the control axis remains."""
    pos = f"{prompt}, {config.control_positive}" if prompt else config.control_positive
    neg = f"{prompt}, {config.control_negative}" if prompt else config.control_negative
    return (score_fn(images, pos) - score_fn(images, neg)) / max(config.reward_temperature, 1e-6)


def timestep_reward_mask(trajectory_fractions: torch.Tensor, *, cutoff: float = 0.75) -> torch.Tensor:
    """1.0 for early/mid trajectory positions (0=noise, 1=clean), 0.0 past ``cutoff`` (Direct-Align guard)."""
    return (trajectory_fractions.float() <= float(cutoff)).float()


__all__ = [
    "FlowGRPOConfig",
    "SRPOConfig",
    "decode_latent_to_rgb_uint8",
    "grpo_weighted_loss",
    "group_relative_advantages",
    "flow_sde_step",
    "reference_kl_penalty",
    "rollout_flow_sample",
    "semantic_relative_reward",
    "timestep_reward_mask",
]
