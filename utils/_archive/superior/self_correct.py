"""
**Self-correction** policy: CLIP alignment gates + optional rewind metadata for multi-pass sampling.

Use with ``sample.py`` refinement flags or a custom loop that decodes latents between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
from utils.generation.inference_research_hooks import RewindState, score_latent_prompt_alignment, should_rewind


@dataclass(slots=True)
class SelfCorrectConfig:
    """Thresholds for alignment-based correction."""

    clip_model_id: str = "openai/clip-vit-base-patch32"
    align_threshold: float = 0.32
    """Mapped CLIP score in [0,1] below this triggers correction."""
    max_refinements: int = 1
    refine_steps: int = 8
    refine_t_frac: float = 0.12
    track_rewinds: bool = True


@dataclass(slots=True)
class SelfCorrectPolicy:
    """Stateful policy across denoise steps or full samples."""

    config: SelfCorrectConfig
    rewind_stack: List[RewindState] = field(default_factory=list)
    refinement_count: int = 0

    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        *,
        vae: Any,
        latent_scale: float,
        ae_type: str = "kl",
        rae_bridge: Any = None,
        device: Optional[torch.device] = None,
    ) -> float:
        dev = device or latent.device
        return score_latent_prompt_alignment(
            latent,
            prompt,
            device=dev,
            clip_model_id=self.config.clip_model_id,
            vae=vae,
            latent_scale=latent_scale,
            ae_type=ae_type,
            rae_bridge=rae_bridge,
        )

    def needs_correction(self, score: float) -> bool:
        if self.refinement_count >= self.config.max_refinements:
            return False
        return should_rewind(score, threshold=self.config.align_threshold)

    def push_rewind(self, latent: torch.Tensor, step_index: int, timestep_value: int) -> None:
        if not self.config.track_rewinds:
            return
        self.rewind_stack.append(
            RewindState(latent=latent.detach().clone(), step_index=step_index, timestep_value=timestep_value)
        )

    def pop_rewind(self) -> Optional[RewindState]:
        if not self.rewind_stack:
            return None
        return self.rewind_stack.pop()

    def note_refinement_done(self) -> None:
        self.refinement_count += 1

    def refine_plan(self, num_timesteps: int) -> tuple[int, int]:
        """Return ``(t_start, num_steps)`` for a short re-denoise pass."""
        t_start = int(float(self.config.refine_t_frac) * max(1, num_timesteps - 1))
        t_start = min(max(1, t_start), num_timesteps - 1)
        return t_start, int(self.config.refine_steps)


__all__ = ["SelfCorrectConfig", "SelfCorrectPolicy"]
