"""
Weathering / patina — rust, wear, repair, graffiti layers as narrative.

Different from era/period: this is *object biography*, not calendar year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class WearLevel(str, Enum):
    PRISTINE = "pristine"
    WORN = "worn"
    WEATHERED = "weathered"
    RUINED = "ruined"
    REPAIRED = "repaired"
    LAYERED = "layered"


@dataclass(frozen=True)
class PatinaPlan:
    level: WearLevel
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, WearLevel, str, str], ...] = (
    (
        re.compile(r"\b(rust|rusted|oxidized|corroded|patina)\b", re.I),
        WearLevel.WEATHERED,
        "honest oxidation, rust streak direction, metal decay at edges",
        "plastic rust paint, uniform orange",
    ),
    (
        re.compile(r"\b(cracked|peeling paint|flaking|sun-bleached)\b", re.I),
        WearLevel.WORN,
        "UV-faded paint, crack network, substrate showing through",
        "random noise cracks",
    ),
    (
        re.compile(r"\b(ruins|collapsed|decay|abandoned|post-apocalyptic)\b", re.I),
        WearLevel.RUINED,
        "structural failure logic, vegetation reclaim, debris scatter",
        "pristine building labeled abandoned",
    ),
    (
        re.compile(r"\b(repaired|patched|duct tape|makeshift fix)\b", re.I),
        WearLevel.REPAIRED,
        "visible repair history, mismatched materials, human fix marks",
        "invisible restoration, museum perfect",
    ),
    (
        re.compile(r"\b(graffiti layers|sticker bomb|wheatpaste|urban decay)\b", re.I),
        WearLevel.LAYERED,
        "overlapping poster tear, graffiti palimpsest, urban time layers",
        "single flat graffiti sticker",
    ),
    (
        re.compile(r"\b(brand new|factory fresh|pristine|mint condition)\b", re.I),
        WearLevel.PRISTINE,
        "unworn edges, clean manufacturing, no patina",
        "scuffs on new object, fake wear",
    ),
)


class PatinaStoryteller:
    def plan(self, prompt: str) -> PatinaPlan:
        text = prompt or ""
        for pat, level, pos, neg in _RULES:
            if pat.search(text):
                return PatinaPlan(level, pos, neg)
        return PatinaPlan(WearLevel.PRISTINE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["PatinaPlan", "PatinaStoryteller", "WearLevel"]
