"""Visual rhythm and pattern beat — music for the eye."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class VisualRhythm(str, Enum):
    REPETITION = "repetition"
    RADIAL = "radial"
    GRID = "grid"
    SPIRAL = "spiral"
    BROKEN = "broken"
    NONE = "none"


@dataclass(frozen=True)
class RhythmPlan:
    rhythm: VisualRhythm
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, VisualRhythm, str, str], ...] = (
    (
        re.compile(r"\b(repeating|pattern|rhythm|motif|series of)\b", re.I),
        VisualRhythm.REPETITION,
        "motif repetition with variation, visual beat every third element",
        "clone stamp spam, identical duplicates",
    ),
    (
        re.compile(r"\b(radial|mandala|sunburst|concentric)\b", re.I),
        VisualRhythm.RADIAL,
        "radial symmetry with organic deviation, center-weighted energy",
        "perfect mathematical mandala filter",
    ),
    (
        re.compile(r"\b(grid|checker|tessellat|tiling pattern)\b", re.I),
        VisualRhythm.GRID,
        "disciplined grid with one broken cell, op-art restraint",
        "warped grid without intent",
    ),
    (
        re.compile(r"\b(spiral|vortex|whirlpool|fibonacci)\b", re.I),
        VisualRhythm.SPIRAL,
        "logarithmic spiral flow, eye travels inward",
        "random swirl smear",
    ),
    (
        re.compile(r"\b(broken rhythm|syncopat|irregular pattern)\b", re.I),
        VisualRhythm.BROKEN,
        "syncopated visual rhythm, deliberate off-beat element",
        "chaotic noise without pattern anchor",
    ),
)


class RhythmPlanner:
    def plan(self, prompt: str) -> RhythmPlan:
        text = prompt or ""
        for pat, rhythm, pos, neg in _RULES:
            if pat.search(text):
                return RhythmPlan(rhythm, pos, neg)
        return RhythmPlan(VisualRhythm.NONE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["RhythmPlan", "RhythmPlanner", "VisualRhythm"]
