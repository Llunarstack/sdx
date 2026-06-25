"""
Composition planner — focal hierarchy, framing, and leading-line vocabulary.

Models default to centered mugshots; this pushes intentional design reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class FramingStyle(str, Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    CENTER_HERO = "center_hero"
    GOLDEN_RATIO = "golden_ratio"
    SYMMETRY = "symmetry"
    DUTCH = "dutch"
    NEGATIVE_SPACE = "negative_space"
    LAYERED_DEPTH = "layered_depth"


@dataclass(frozen=True)
class CompositionPlan:
    style: FramingStyle
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, FramingStyle, str, str], ...] = (
    (
        re.compile(r"\b(rule of thirds|off[- ]center|thirds composition)\b", re.I),
        FramingStyle.RULE_OF_THIRDS,
        "subject on thirds intersection, balanced asymmetry, breathing room in frame",
        "dead center snapshot, cramped composition",
    ),
    (
        re.compile(r"\b(symmetr|reflection|mirror|centered|symmetrical)\b", re.I),
        FramingStyle.SYMMETRY,
        "axial symmetry, mirrored balance, strong central axis",
        "accidental asymmetry drift, lopsided frame",
    ),
    (
        re.compile(r"\b(dutch angle|tilted|diagonal tension|canted)\b", re.I),
        FramingStyle.DUTCH,
        "controlled dutch tilt for tension, dynamic diagonal energy",
        "random skew without intent, horizon confusion",
    ),
    (
        re.compile(r"\b(minimal|negative space|lots of empty|breathing room)\b", re.I),
        FramingStyle.NEGATIVE_SPACE,
        "generous negative space, subject isolation, calm visual rest",
        "cluttered frame, every pixel filled with noise",
    ),
    (
        re.compile(r"\b(foreground|midground|background|layers|depth layers)\b", re.I),
        FramingStyle.LAYERED_DEPTH,
        "foreground framing element, clear midground subject, atmospheric background separation",
        "flat cardboard cutout depth, single plane mush",
    ),
    (
        re.compile(r"\b(hero shot|centered subject|iconic pose)\b", re.I),
        FramingStyle.CENTER_HERO,
        "strong centered hero read, symmetrical weight, poster clarity",
        "weak off-center drift without purpose",
    ),
)


class CompositionPlanner:
    def plan(self, prompt: str) -> CompositionPlan:
        text = prompt or ""
        for pat, style, pos, neg in _RULES:
            if pat.search(text):
                return CompositionPlan(style, pos, neg)
        if re.search(r"\b(portrait|landscape|cityscape|still life)\b", text, re.I):
            return CompositionPlan(
                FramingStyle.RULE_OF_THIRDS,
                "clear focal hierarchy, intentional framing, subject separation from background",
                "muddy composition, no clear focal point",
            )
        return CompositionPlan(FramingStyle.RULE_OF_THIRDS, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["CompositionPlan", "CompositionPlanner", "FramingStyle"]
