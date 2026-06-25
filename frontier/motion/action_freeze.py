"""Motion blur direction and action peak moments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class MotionStyle(str, Enum):
    FREEZE = "freeze"
    PAN = "pan"
    TRACK = "track"
    SPIN = "spin"
    IMPACT = "impact"
    FALL = "fall"


@dataclass(frozen=True)
class MotionPlan:
    style: MotionStyle
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, MotionStyle, str, str], ...] = (
    (
        re.compile(r"\b(motion blur|panning shot|speed| racing|running)\b", re.I),
        MotionStyle.PAN,
        "directional motion blur on background, sharp subject core, panning streaks",
        "global blur mush, subject and background equally smeared",
    ),
    (
        re.compile(r"\b(frozen|mid[- ]air|peak action|decisive moment|splash)\b", re.I),
        MotionStyle.FREEZE,
        "frozen peak action, crisp subject edges, suspended particles",
        "motion smear on static pose, ambiguous timing",
    ),
    (
        re.compile(r"\b(punch|impact|explosion|collision|shatter)\b", re.I),
        MotionStyle.IMPACT,
        "radial debris, impact frame read, force direction clear",
        "random particles without force vector",
    ),
    (
        re.compile(r"\b(falling|jump|leap| dive)\b", re.I),
        MotionStyle.FALL,
        "gravity-readable pose, hair and cloth lag, vertical motion cue",
        "floating without weight, stiff cloth",
    ),
)


class MotionPlanner:
    def plan(self, prompt: str) -> MotionPlan:
        text = prompt or ""
        for pat, style, pos, neg in _RULES:
            if pat.search(text):
                return MotionPlan(style, pos, neg)
        return MotionPlan(MotionStyle.FREEZE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["MotionPlan", "MotionPlanner", "MotionStyle"]
