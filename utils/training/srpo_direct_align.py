"""
**SRPO / Direct-Align** scaffold — trajectory-level preference training that
fixed the "waxy Flux skin" problem (Tencent SRPO: Semantic Relative Preference
Optimization, arXiv 2509; "Directly Aligning the Full Diffusion Trajectory").

Two ideas, both usable independently:

1. **Direct-Align recovery** — diffusion states are interpolations between a
   *known* noise prior and the clean image, so the clean image can be recovered
   from any timestep in one step (no full rollout). Rewarding these recoveries
   at early/mid timesteps avoids the late-timestep over-optimization that makes
   reward-tuned models glossy.
2. **Semantic relative reward** — instead of an absolute score, reward the
   *difference* between the image scored under a positive control phrase
   ("natural skin texture, realistic photo") and a negative one ("oily glossy
   skin, ai rendered"). Steers an off-the-shelf reward model toward a specific
   axis without retraining it.

Companion to ``utils/training/flow_grpo.py``; same practical-scaffold contract
(single-GPU loop building blocks, not a paper reproduction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = [
    "SRPOConfig",
    "direct_align_recover_flow",
    "direct_align_recover_vp",
    "semantic_relative_reward",
    "srpo_advantage_weights",
    "timestep_reward_mask",
]


@dataclass(slots=True)
class SRPOConfig:
    control_positive: str = "natural skin texture, realistic photograph, true color"
    control_negative: str = "oily glossy skin, waxy plastic render, ai generated look"
    late_timestep_cutoff: float = 0.75
    """Fraction of the trajectory (0=noise, 1=clean) after which rewards are
    masked out — rewarding late steps is what over-polishes skin."""
    reward_temperature: float = 1.0
    advantage_clip: float = 3.0


def direct_align_recover_flow(x_s: torch.Tensor, v_pred: torch.Tensor, s: float) -> torch.Tensor:
    """
    One-step clean-image recovery on a rectified-flow trajectory.

    With ``x_s = (1-s)·x0 + s·ε`` and ``v = ε - x0``: ``x0 = x_s - s·v``.
    """
    return x_s - float(s) * v_pred


def direct_align_recover_vp(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """
    One-step x0 recovery for VP diffusion: ``x0 = (x_t - σ_t·ε̂) / √ᾱ_t``.
    """
    ab = alpha_bar_t.clamp(1e-8, 1.0)
    while ab.ndim < x_t.ndim:
        ab = ab.unsqueeze(-1)
    sigma = (1.0 - ab).clamp(min=0.0).sqrt()
    return (x_t - sigma * eps_pred) / ab.sqrt()


def semantic_relative_reward(
    score_fn: Callable[[torch.Tensor, str], torch.Tensor],
    images: torch.Tensor,
    prompt: str,
    *,
    config: SRPOConfig | None = None,
) -> torch.Tensor:
    """
    Relative preference along the control axis: ``r = score(prompt + positive)
    - score(prompt + negative)``.

    ``score_fn(images, text) -> (B,)`` can be any differentiable or black-box
    reward (CLIP similarity, HPS, ``OnlineRewardModel`` adapter). The
    subtraction cancels the reward model's global aesthetic bias — the exact
    bias that produced the shared "AI look" across frontier models.
    """
    cfg = config or SRPOConfig()
    pos_text = f"{prompt}, {cfg.control_positive}" if prompt else cfg.control_positive
    neg_text = f"{prompt}, {cfg.control_negative}" if prompt else cfg.control_negative
    r_pos = score_fn(images, pos_text)
    r_neg = score_fn(images, neg_text)
    return (r_pos - r_neg) / max(cfg.reward_temperature, 1e-6)


def timestep_reward_mask(trajectory_fractions: torch.Tensor, *, cutoff: float = 0.75) -> torch.Tensor:
    """
    1.0 for early/mid trajectory positions, 0.0 past ``cutoff`` (fraction of
    the way to the clean image). Direct-Align's guard against late-step
    over-optimization.
    """
    return (trajectory_fractions.float() <= float(cutoff)).float()


def srpo_advantage_weights(
    rewards: torch.Tensor,
    trajectory_fractions: torch.Tensor,
    *,
    config: SRPOConfig | None = None,
) -> torch.Tensor:
    """
    Per-sample loss weights: group-normalized relative rewards, clipped, with
    late-timestep contributions masked to zero.

    Multiply into the per-sample denoising loss (same contract as
    ``flow_grpo.grpo_weighted_loss`` weighting).
    """
    cfg = config or SRPOConfig()
    r = rewards.detach().float()
    if r.numel() > 1:
        r = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
    clip = float(cfg.advantage_clip)
    r = r.clamp(-clip, clip)
    mask = timestep_reward_mask(trajectory_fractions, cutoff=cfg.late_timestep_cutoff)
    # exp(-adv): high relative reward → low loss weight (reinforce), matching
    # the flow_grpo convention.
    return torch.exp(-r / max(clip, 1e-6)) * mask
