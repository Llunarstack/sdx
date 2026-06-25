"""Many people — diversity and staging, not duplicate subjects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class CrowdScale(str, Enum):
    SMALL_GROUP = "small_group"
    CROWD = "crowd"
    MASS = "mass"
    NONE = "none"


@dataclass(frozen=True)
class CrowdPlan:
    scale: CrowdScale
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, CrowdScale, str, str], ...] = (
    (
        re.compile(r"\b(crowd|packed|stadium|concert audience|protest march)\b", re.I),
        CrowdScale.CROWD,
        "varied faces and silhouettes, staggered depth rows, no identical clones",
        "duplicate faces, clone army, same person repeated",
    ),
    (
        re.compile(r"\b(two people|couple|pair|duo|three friends)\b", re.I),
        CrowdScale.SMALL_GROUP,
        "distinct individuals, separate styling, clear spatial relationship",
        "merged bodies, twin duplication",
    ),
    (
        re.compile(r"\b(army|horde|thousands|sea of people|massive crowd)\b", re.I),
        CrowdScale.MASS,
        "mass read via silhouette tiers, atmospheric perspective, individual only in foreground",
        "every face sharp and identical",
    ),
)


class CrowdGrammar:
    def plan(self, prompt: str) -> CrowdPlan:
        text = prompt or ""
        for pat, scale, pos, neg in _RULES:
            if pat.search(text):
                return CrowdPlan(scale, pos, neg)
        if re.search(r"\b(people|group|family|team)\b", text, re.I):
            return CrowdPlan(
                CrowdScale.SMALL_GROUP,
                "each person visually distinct, natural group staging",
                "duplicate subjects, extra heads",
            )
        return CrowdPlan(CrowdScale.NONE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["CrowdGrammar", "CrowdPlan", "CrowdScale"]
