"""FRICTION — Anti-plastic microtexture schedule.

Problem: oversmoothed "AI skin". FRICTION adds prompt priors + a late-step
noise scale hint that restores film grain / pore / fabric weave pressure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["FrictionSchedule", "apply_friction_to_prompts", "friction_noise_scale"]


@dataclass
class FrictionSchedule:
    strength: float = 0.55  # 0..1
    late_start: float = 0.65
    peak_noise: float = 0.04

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_POS = (
    "natural skin pores",
    "subtle film grain",
    "asymmetric detail",
    "fabric weave visible",
    "imperfect real-world texture",
    "lived-in surfaces",
)
_NEG = (
    "plastic skin",
    "waxy skin",
    "oversmooth",
    "airbrushed",
    "porcelain doll skin",
    "CGI sheen",
    "beauty filter blur",
)


def apply_friction_to_prompts(
    positive: str,
    negative: str = "",
    schedule: FrictionSchedule | None = None,
) -> tuple[str, str]:
    s = schedule or FrictionSchedule()
    if s.strength <= 0:
        return positive, negative
    k = max(1, int(round(len(_POS) * float(s.strength))))
    pos = str(positive or "").rstrip(",")
    neg = str(negative or "").rstrip(",")
    add_p = ", ".join(_POS[:k])
    add_n = ", ".join(_NEG[: max(2, k)])
    if add_p.lower() not in pos.lower():
        pos = f"{pos}, {add_p}"
    neg = f"{neg}, {add_n}".strip(", ")
    return pos.strip(", "), neg.strip(", ")


def friction_noise_scale(progress: float, schedule: FrictionSchedule | None = None) -> float:
    """Extra step noise multiplier contribution for late denoise (caller adds to η)."""
    s = schedule or FrictionSchedule()
    p = float(progress)
    if p < s.late_start or s.strength <= 0:
        return 0.0
    t = (p - s.late_start) / max(1e-6, 1.0 - s.late_start)
    return float(s.peak_noise * s.strength * t)
