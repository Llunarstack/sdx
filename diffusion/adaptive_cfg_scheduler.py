"""
Adaptive CFG scheduler: learns the optimal per-step CFG scale for each generation.

Standard CFG uses a fixed scale (e.g. 7.5) for all steps. This is suboptimal:
- Early steps (high noise): high CFG helps establish correct composition
- Mid steps: moderate CFG balances structure and diversity
- Late steps (low noise): lower CFG prevents oversaturation and artifacts

This module provides:
1. CosineAnnealingCFG — simple cosine schedule (no learning needed)
2. PromptComplexityCFG — adapts CFG based on prompt complexity
3. LatentDeltaCFG — adapts CFG based on how much the latent is changing
4. LearnedCFGSchedule — a small MLP that predicts optimal CFG from
   (timestep, latent_stats, prompt_embedding_norm) — can be trained
   with DPO or reward signal
5. HybridAdaptiveCFG — combines all of the above

The key insight: CFG is not a single number, it's a function of:
- How noisy the current latent is (timestep)
- How well the generation is tracking the prompt (CLIP score proxy)
- How complex the prompt is (number of concepts, conflicts)
- How much the latent is changing (stability signal)

Usage in sample.py:
    scheduler = HybridAdaptiveCFG(
        base_cfg=7.5,
        prompt=prompt,
        total_steps=50,
    )
    # In sampling loop:
    cfg_this_step = scheduler.get_cfg(step_i, x_t, t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Simple schedule-based CFG
# ---------------------------------------------------------------------------


class CosineAnnealingCFG:
    """
    Cosine annealing CFG schedule.

    Starts at cfg_max, anneals to cfg_min following a cosine curve.
    Optionally has a warmup phase where CFG ramps up first.

    This alone often improves quality over fixed CFG:
    - High CFG early: establishes correct composition and subject
    - Lower CFG late: prevents oversaturation and edge artifacts
    """

    def __init__(
        self,
        cfg_max: float = 9.0,
        cfg_min: float = 5.0,
        total_steps: int = 50,
        warmup_frac: float = 0.1,
        peak_frac: float = 0.3,
    ):
        self.cfg_max = float(cfg_max)
        self.cfg_min = float(cfg_min)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_frac * total_steps)
        self.peak_step = int(peak_frac * total_steps)

    def get_cfg(self, step: int, **kwargs) -> float:
        """Get CFG scale for the current step."""
        if step < self.warmup_steps:
            # Linear warmup
            t = step / max(self.warmup_steps, 1)
            return self.cfg_min + t * (self.cfg_max - self.cfg_min)
        elif step <= self.peak_step:
            # Hold at max
            return self.cfg_max
        else:
            # Cosine anneal from max to min
            progress = (step - self.peak_step) / max(self.total_steps - self.peak_step, 1)
            cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.cfg_min + cos_val * (self.cfg_max - self.cfg_min)


class LinearDecayCFG:
    """Linear CFG decay from cfg_start to cfg_end."""

    def __init__(self, cfg_start: float = 9.0, cfg_end: float = 5.0, total_steps: int = 50):
        self.cfg_start = float(cfg_start)
        self.cfg_end = float(cfg_end)
        self.total_steps = int(total_steps)

    def get_cfg(self, step: int, **kwargs) -> float:
        t = min(1.0, step / max(self.total_steps - 1, 1))
        return self.cfg_start + t * (self.cfg_end - self.cfg_start)


class StepwiseCFG:
    """
    Stepwise CFG: different values for different phases.

    Phase 1 (0-30%): High CFG for composition
    Phase 2 (30-70%): Medium CFG for structure
    Phase 3 (70-100%): Lower CFG for detail/texture
    """

    def __init__(
        self,
        cfg_phase1: float = 9.0,
        cfg_phase2: float = 7.5,
        cfg_phase3: float = 5.5,
        total_steps: int = 50,
    ):
        self.phases = [
            (0.30, cfg_phase1),
            (0.70, cfg_phase2),
            (1.00, cfg_phase3),
        ]
        self.total_steps = int(total_steps)

    def get_cfg(self, step: int, **kwargs) -> float:
        frac = step / max(self.total_steps - 1, 1)
        for threshold, cfg in self.phases:
            if frac <= threshold:
                return cfg
        return self.phases[-1][1]


# ---------------------------------------------------------------------------
# Prompt-complexity-aware CFG
# ---------------------------------------------------------------------------


class PromptComplexityCFG:
    """
    Adapts CFG based on prompt complexity.

    Complex prompts (many concepts, conflicts, unusual combinations) need
    higher CFG to ensure all elements are represented.
    Simple prompts can use lower CFG for more natural results.
    """

    def __init__(
        self,
        base_cfg: float = 7.5,
        complexity_boost: float = 2.0,
        total_steps: int = 50,
    ):
        self.base_cfg = float(base_cfg)
        self.complexity_boost = float(complexity_boost)
        self.total_steps = int(total_steps)
        self._complexity: float = 0.5  # default

    def set_prompt(self, prompt: str) -> None:
        """Analyze prompt and set complexity score."""
        self._complexity = self._analyze_complexity(prompt)

    def _analyze_complexity(self, prompt: str) -> float:
        """Estimate prompt complexity from 0 (simple) to 1 (very complex)."""
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        n_parts = len(parts)

        # Factors:
        # 1. Number of comma-separated elements
        part_score = min(1.0, n_parts / 15.0)

        # 2. Presence of spatial relationships (hard for models)
        spatial_words = ["behind", "in front of", "next to", "above", "below", "between", "left", "right"]
        spatial_score = min(1.0, sum(1 for w in spatial_words if w in prompt.lower()) / 3.0)

        # 3. Multiple subjects
        subject_words = ["and", "with", "alongside", "together", "couple", "group", "two", "three"]
        multi_score = min(1.0, sum(1 for w in subject_words if w in prompt.lower()) / 2.0)

        # 4. Unusual combinations (style mixing)
        style_words = ["realistic", "anime", "3d", "painting", "photo", "illustration"]
        style_count = sum(1 for w in style_words if w in prompt.lower())
        style_score = min(1.0, max(0.0, style_count - 1) / 2.0)

        complexity = 0.4 * part_score + 0.25 * spatial_score + 0.2 * multi_score + 0.15 * style_score
        return float(min(1.0, complexity))

    def get_cfg(self, step: int, **kwargs) -> float:
        """Get CFG scale, boosted for complex prompts."""
        # Apply cosine decay on top of complexity boost
        t = step / max(self.total_steps - 1, 1)
        decay = 0.5 * (1.0 + math.cos(math.pi * t * 0.7))  # gentle decay

        complexity_boost = self._complexity * self.complexity_boost * decay
        return self.base_cfg + complexity_boost


# ---------------------------------------------------------------------------
# Latent-delta-based adaptive CFG
# ---------------------------------------------------------------------------


class LatentDeltaCFG:
    """
    Adapts CFG based on how much the latent is changing.

    If the latent is changing a lot (high delta), the generation is still
    evolving rapidly — we can afford lower CFG.
    If the latent is barely changing (low delta), the generation may be
    stuck — boost CFG to push it toward the prompt.

    This is a more principled version of --volatile-cfg-boost.
    """

    def __init__(
        self,
        base_cfg: float = 7.5,
        boost_when_stuck: float = 1.5,
        stuck_threshold: float = 0.01,
        window: int = 5,
        total_steps: int = 50,
    ):
        self.base_cfg = float(base_cfg)
        self.boost_when_stuck = float(boost_when_stuck)
        self.stuck_threshold = float(stuck_threshold)
        self.window = int(window)
        self.total_steps = int(total_steps)
        self._delta_history: List[float] = []
        self._current_boost: float = 1.0

    def update(self, latent: torch.Tensor, prev_latent: Optional[torch.Tensor]) -> None:
        """Update delta history with the current latent change."""
        if prev_latent is None:
            return
        with torch.no_grad():
            delta = float((latent - prev_latent).abs().mean().item())
        self._delta_history.append(delta)
        if len(self._delta_history) > self.window:
            self._delta_history.pop(0)

        # Compute boost
        if len(self._delta_history) >= 3:
            recent_delta = sum(self._delta_history[-3:]) / 3
            if recent_delta < self.stuck_threshold:
                # Stuck: boost CFG
                self._current_boost = self.boost_when_stuck
            else:
                # Moving: decay boost back to 1
                self._current_boost = max(1.0, self._current_boost * 0.9)

    def get_cfg(self, step: int, **kwargs) -> float:
        return self.base_cfg * self._current_boost


# ---------------------------------------------------------------------------
# Learned CFG schedule (trainable MLP)
# ---------------------------------------------------------------------------


class LearnedCFGSchedule(nn.Module):
    """
    Small MLP that predicts optimal CFG from generation state.

    Inputs:
    - Normalized timestep t ∈ [0, 1]
    - Latent statistics (mean, std, skewness)
    - Prompt embedding norm (proxy for prompt complexity)
    - Step fraction (how far through generation)

    Output: CFG scale multiplier ∈ [0.5, 2.0]

    Training: use DPO or reward signal to train this MLP to predict
    the CFG that maximizes image quality for each generation state.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        base_cfg: float = 7.5,
        cfg_min: float = 3.0,
        cfg_max: float = 15.0,
    ):
        super().__init__()
        self.base_cfg = float(base_cfg)
        self.cfg_min = float(cfg_min)
        self.cfg_max = float(cfg_max)

        # Input: [t_norm, latent_mean, latent_std, latent_skew, prompt_norm, step_frac] = 6 dims
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

        # Initialize to output ~0.5 (maps to base_cfg)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(
        self,
        t_norm: float,
        latent: Optional[torch.Tensor] = None,
        prompt_emb_norm: float = 1.0,
        step_frac: float = 0.5,
    ) -> float:
        """
        Predict optimal CFG for the current generation state.

        Args:
            t_norm: Normalized timestep in [0, 1]
            latent: Current latent (B, C, H, W) for statistics
            prompt_emb_norm: L2 norm of prompt embedding (proxy for complexity)
            step_frac: Fraction of total steps completed

        Returns:
            CFG scale value
        """
        # Compute latent statistics
        if latent is not None:
            with torch.no_grad():
                z = latent.float()
                lat_mean = float(z.mean().item())
                lat_std = float(z.std().item())
                # Skewness proxy: mean of cubed normalized values
                z_norm = (z - z.mean()) / (z.std() + 1e-8)
                lat_skew = float(z_norm.pow(3).mean().item())
        else:
            lat_mean, lat_std, lat_skew = 0.0, 1.0, 0.0

        # Normalize inputs
        features = torch.tensor(
            [
                float(t_norm),
                float(lat_mean) / 4.0,  # typical range [-4, 4]
                float(lat_std) / 2.0,  # typical range [0, 2]
                float(lat_skew) / 2.0,  # typical range [-2, 2]
                float(prompt_emb_norm) / 100.0,  # normalize
                float(step_frac),
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            output = self.net(features).item()  # [0, 1]

        # Map to CFG range
        cfg = self.cfg_min + output * (self.cfg_max - self.cfg_min)
        return float(cfg)


# ---------------------------------------------------------------------------
# Hybrid adaptive CFG (combines all strategies)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdaptiveCFGConfig:
    """Configuration for HybridAdaptiveCFG."""

    base_cfg: float = 7.5
    use_cosine_schedule: bool = True
    use_complexity_boost: bool = True
    use_latent_delta: bool = True
    use_learned: bool = False  # Requires trained MLP
    cosine_cfg_max: float = 9.0
    cosine_cfg_min: float = 5.5
    complexity_boost: float = 1.5
    stuck_boost: float = 1.3
    stuck_threshold: float = 0.008
    total_steps: int = 50
    prompt: str = ""
    learned_ckpt: Optional[str] = None  # Path to trained LearnedCFGSchedule


class HybridAdaptiveCFG:
    """
    Hybrid adaptive CFG that combines multiple strategies.

    This is the recommended entry point. It:
    1. Applies a cosine schedule (always)
    2. Boosts for complex prompts (if enabled)
    3. Boosts when generation is stuck (if enabled)
    4. Optionally uses a learned MLP for fine-grained control

    The final CFG is a weighted combination of all active strategies.
    """

    def __init__(self, cfg: Optional[AdaptiveCFGConfig] = None, **kwargs):
        if cfg is None:
            cfg = AdaptiveCFGConfig(**{k: v for k, v in kwargs.items() if hasattr(AdaptiveCFGConfig, k)})
        self.cfg = cfg

        self.cosine = (
            CosineAnnealingCFG(
                cfg_max=cfg.cosine_cfg_max,
                cfg_min=cfg.cosine_cfg_min,
                total_steps=cfg.total_steps,
            )
            if cfg.use_cosine_schedule
            else None
        )

        self.complexity = (
            PromptComplexityCFG(
                base_cfg=cfg.base_cfg,
                complexity_boost=cfg.complexity_boost,
                total_steps=cfg.total_steps,
            )
            if cfg.use_complexity_boost
            else None
        )

        if self.complexity and cfg.prompt:
            self.complexity.set_prompt(cfg.prompt)

        self.delta = (
            LatentDeltaCFG(
                base_cfg=cfg.base_cfg,
                boost_when_stuck=cfg.stuck_boost,
                stuck_threshold=cfg.stuck_threshold,
                total_steps=cfg.total_steps,
            )
            if cfg.use_latent_delta
            else None
        )

        self.learned: Optional[LearnedCFGSchedule] = None
        if cfg.use_learned and cfg.learned_ckpt:
            try:
                self.learned = LearnedCFGSchedule()
                state = torch.load(cfg.learned_ckpt, map_location="cpu", weights_only=True)
                self.learned.load_state_dict(state)
                self.learned.eval()
            except Exception:
                self.learned = None

        self._prev_latent: Optional[torch.Tensor] = None
        self._history: List[float] = []

    def update_latent(self, latent: torch.Tensor) -> None:
        """Update with current latent for delta tracking."""
        if self.delta is not None:
            self.delta.update(latent, self._prev_latent)
        self._prev_latent = latent.detach()

    def get_cfg(
        self,
        step: int,
        latent: Optional[torch.Tensor] = None,
        t_norm: Optional[float] = None,
        prompt_emb_norm: float = 1.0,
    ) -> float:
        """
        Get the adaptive CFG scale for the current step.

        Args:
            step: Current step index (0-based)
            latent: Current latent (for delta tracking and learned schedule)
            t_norm: Normalized timestep in [0, 1] (if None, computed from step)
            prompt_emb_norm: L2 norm of prompt embedding

        Returns:
            CFG scale value
        """
        if t_norm is None:
            t_norm = 1.0 - step / max(self.cfg.total_steps - 1, 1)

        step_frac = step / max(self.cfg.total_steps - 1, 1)

        # Collect CFG values from each active strategy
        cfgs = []
        weights = []

        if self.cosine is not None:
            cfgs.append(self.cosine.get_cfg(step))
            weights.append(0.5)

        if self.complexity is not None:
            cfgs.append(self.complexity.get_cfg(step))
            weights.append(0.3)

        if self.delta is not None:
            if latent is not None:
                self.delta.update(latent, self._prev_latent)
                self._prev_latent = latent.detach()
            cfgs.append(self.delta.get_cfg(step))
            weights.append(0.2)

        if self.learned is not None and latent is not None:
            learned_cfg = self.learned(t_norm, latent, prompt_emb_norm, step_frac)
            cfgs.append(learned_cfg)
            weights.append(0.4)

        if not cfgs:
            return self.cfg.base_cfg

        # Weighted average
        total_weight = sum(weights)
        cfg_value = sum(c * w for c, w in zip(cfgs, weights)) / total_weight

        # Clamp to reasonable range
        cfg_value = max(1.0, min(20.0, cfg_value))
        self._history.append(cfg_value)

        return cfg_value

    def get_history(self) -> List[float]:
        """Return the CFG values used across all steps."""
        return list(self._history)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the CFG schedule used."""
        if not self._history:
            return {}
        return {
            "mean_cfg": sum(self._history) / len(self._history),
            "max_cfg": max(self._history),
            "min_cfg": min(self._history),
            "n_steps": len(self._history),
            "history": self._history,
        }


__all__ = [
    "HybridAdaptiveCFG",
    "AdaptiveCFGConfig",
    "CosineAnnealingCFG",
    "LinearDecayCFG",
    "StepwiseCFG",
    "PromptComplexityCFG",
    "LatentDeltaCFG",
    "LearnedCFGSchedule",
]
