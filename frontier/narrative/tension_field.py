"""
Tension Field — maps narrative tension to diffusion step emphasis (SDX image / keyframe).

Unlike static CFG, high-tension prompts get mid-step structure bursts;
low-tension prompts lock composition early and texture late.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["TensionField", "TensionFieldPlan", "build_tension_field"]


@dataclass(frozen=True)
class TensionFieldPlan:
    tension: float
    prompt_fragments: Tuple[str, ...]
    step_emphasis: Tuple[float, ...]
    cfg_boost: float


@dataclass
class TensionField:
    num_steps: int = 28

    def plan(self, tension: float) -> TensionFieldPlan:
        t = max(0.0, min(1.0, float(tension)))
        n = max(4, self.num_steps)
        if t >= 0.75:
            frags = ("sharp micro-contrast", "urgent diagonal energy", "high-stakes focal sharpness")
            curve = tuple(0.55 + 0.9 * (1.0 - abs(i / max(1, n - 1) - 0.35)) for i in range(n))
            cfg = 1.0 + t * 0.35
        elif t >= 0.4:
            frags = ("controlled drama", "building visual pressure")
            curve = tuple(0.7 + 0.5 * (i / max(1, n - 1)) for i in range(n))
            cfg = 1.0 + t * 0.15
        else:
            frags = ("calm negative space", "gentle atmospheric falloff")
            curve = tuple(1.0 - 0.55 * (i / max(1, n - 1)) for i in range(n))
            cfg = 1.0
        return TensionFieldPlan(tension=t, prompt_fragments=frags, step_emphasis=curve, cfg_boost=cfg)


def build_tension_field(num_steps: int = 28) -> TensionField:
    return TensionField(num_steps=num_steps)
