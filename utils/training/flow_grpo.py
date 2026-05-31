"""
**Flow-GRPO** scaffold for online RL on flow-matching / VP DiT models.

Inspired by Flow-GRPO (NeurIPS 2025): ODE→SDE noise injection for exploration,
terminal reward from ``OnlineRewardModel``, group-relative policy update on
denoising surrogate loss.

This is a **practical SDX trainer** — not a full reproduction of multi-GPU GRPO,
but a working loop: sample → score → weighted fine-tune on high-reward trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


def inject_sde_noise(
    x: torch.Tensor,
    v_pred: torch.Tensor,
    dt: float,
    *,
    noise_scale: float = 0.35,
) -> torch.Tensor:
    """
    SDE-style Euler step with injected Gaussian noise (exploration for GRPO).

    ``x_next = x + v * dt + sqrt(dt) * noise_scale * eps``
    """
    dt_f = float(max(dt, 1e-6))
    eps = torch.randn_like(x)
    return x + v_pred * dt_f + (dt_f**0.5) * float(noise_scale) * eps


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
) -> torch.Tensor:
    """
    Weight per-sample denoising loss by normalized advantages (higher reward → lower weight on loss reduction).

    Policy wants to **minimize** loss on high-advantage samples → we use ``loss * exp(-adv/clip)``.
    """
    adv = advantages.detach().float().clamp(-float(clip), float(clip))
    w = torch.exp(-adv / max(float(clip), 1e-6))
    return (per_sample_loss * w).sum() / (w.sum() + 1e-8)


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
    from utils.generation.latent_edit_helpers import vae_decode_dit_latent_to_tensor01

    img01 = vae_decode_dit_latent_to_tensor01(
        vae, latent, latent_scale=float(latent_scale), ae_type=str(ae_type), rae_bridge=rae_bridge
    )
    arr = (img01[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def rollout_flow_sample(
    model: torch.nn.Module,
    diffusion: Any,
    shape: Tuple[int, ...],
    *,
    model_kwargs_cond: Dict[str, Any],
    model_kwargs_uncond: Optional[Dict[str, Any]],
    cfg_scale: float,
    steps: int,
    device: str,
    sde_noise_scale: float = 0.35,
) -> torch.Tensor:
    """
    Short flow-matching rollout with optional SDE noise between steps.

    Falls back to standard ``sample_loop`` when flow matching is unavailable.
    """
    if hasattr(diffusion, "sample_loop"):
        return diffusion.sample_loop(
            model,
            shape,
            model_kwargs_cond=model_kwargs_cond,
            model_kwargs_uncond=model_kwargs_uncond,
            cfg_scale=float(cfg_scale),
            num_inference_steps=max(2, int(steps)),
            device=device,
            dtype=torch.float32,
            flow_matching_sample=True,
            flow_solver="euler",
        )
    raise RuntimeError("diffusion.sample_loop required for rollout")


__all__ = [
    "FlowGRPOConfig",
    "decode_latent_to_rgb_uint8",
    "grpo_weighted_loss",
    "group_relative_advantages",
    "inject_sde_noise",
    "reference_kl_penalty",
    "rollout_flow_sample",
]
