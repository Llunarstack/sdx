"""
Dream logic — when the prompt wants surrealism, stop fighting it.

``logic/contradiction`` resolves conflicts; this module *amplifies* intentional impossibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class SurrealMode(str, Enum):
    NONE = "none"
    DREAM = "dream"
    METAMORPHOSIS = "metamorphosis"
    SCALE_SHIFT = "scale_shift"
    FLOATING = "floating"
    MELTING = "melting"
    VAST_TINY = "vast_tiny"


@dataclass(frozen=True)
class DreamLogicPlan:
    mode: SurrealMode
    positive: str
    negative: str
    serendipity_boost: float  # add to base dial


_RULES: Tuple[Tuple[re.Pattern, SurrealMode, str, str, float], ...] = (
    (
        re.compile(r"\b(surreal|surrealism|dreamlike|dreamscape|oneiric)\b", re.I),
        SurrealMode.DREAM,
        "oneiric logic, symbolic juxtaposition, soft impossible transitions",
        "literal stock photo realism, mundane documentary",
        0.15,
    ),
    (
        re.compile(r"\b(morph|metamorph|transforming into|becoming)\b", re.I),
        SurrealMode.METAMORPHOSIS,
        "readable in-between form, coherent metamorphosis midpoint, intentional hybrid silhouette",
        "hard cut paste collage, incoherent limb soup",
        0.2,
    ),
    (
        re.compile(r"\b(giant tiny|colossal|miniature person|wrong scale)\b", re.I),
        SurrealMode.SCALE_SHIFT,
        "deliberate scale paradox, clear size relationship, toy-world or titan-world read",
        "accidental scale drift without intent",
        0.18,
    ),
    (
        re.compile(r"\b(floating|levitat|anti[- ]gravity|weightless)\b", re.I),
        SurrealMode.FLOATING,
        "weightless suspension with shadow or tether cue, dream physics",
        "random float without composition reason",
        0.12,
    ),
    (
        re.compile(r"\b(melting|dali|soft clock|viscous form)\b", re.I),
        SurrealMode.MELTING,
        "viscous deformation along gravity suggestion, molten form rhythm",
        "jpeg smear, random liquify artifact",
        0.2,
    ),
    (
        re.compile(r"\b(vast tiny|infinite|endless|impossibly large)\b", re.I),
        SurrealMode.VAST_TINY,
        "awe scale contrast, tiny figure against immense architecture or void",
        "flat backdrop without depth",
        0.1,
    ),
)


class DreamLogicPlanner:
    def plan(self, prompt: str) -> DreamLogicPlan:
        text = prompt or ""
        for pat, mode, pos, neg, boost in _RULES:
            if pat.search(text):
                return DreamLogicPlan(mode, pos, neg, boost)
        return DreamLogicPlan(SurrealMode.NONE, "", "", 0.0)

    def fragments(self, prompt: str) -> Tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.serendipity_boost


__all__ = ["DreamLogicPlan", "DreamLogicPlanner", "SurrealMode"]
