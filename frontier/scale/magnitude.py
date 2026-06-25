"""Scale relationships — make size contrast readable and intentional."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ScaleRelation(str, Enum):
    TITAN = "titan"
    MINIATURE = "miniature"
    COSMIC = "cosmic"
    INTIMATE = "intimate"
    NONE = "none"


@dataclass(frozen=True)
class MagnitudePlan:
    relation: ScaleRelation
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, ScaleRelation, str, str], ...] = (
    (
        re.compile(r"\b(colossal|giant|titan|kaiju|massive creature|towering)\b", re.I),
        ScaleRelation.TITAN,
        "scale cue via tiny foreground reference, atmospheric haze on upper mass",
        "giant without scale reference, floating size",
    ),
    (
        re.compile(r"\b(miniature|tiny world|microcosm|diorama scale)\b", re.I),
        ScaleRelation.MINIATURE,
        "forced perspective miniature read, shallow depth, toy-like material honesty",
        "real scale objects in miniature scene",
    ),
    (
        re.compile(r"\b(cosmic|galaxy|planet|astronomical|nebula backdrop)\b", re.I),
        ScaleRelation.COSMIC,
        "figure against cosmic scale, rim light from nebula, humility framing",
        "studio backdrop pretending to be space",
    ),
    (
        re.compile(r"\b(intimate scale|close world|macro universe|hands only)\b", re.I),
        ScaleRelation.INTIMATE,
        "micro-narrative in small frame, tactile nearness, world implied off-frame",
        "wide shot when intimacy requested",
    ),
)


class MagnitudePlanner:
    def plan(self, prompt: str) -> MagnitudePlan:
        text = prompt or ""
        for pat, rel, pos, neg in _RULES:
            if pat.search(text):
                return MagnitudePlan(rel, pos, neg)
        return MagnitudePlan(ScaleRelation.NONE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["MagnitudePlan", "MagnitudePlanner", "ScaleRelation"]
