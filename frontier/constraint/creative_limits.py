"""
Creative constraints — art-school limits that beat "add more detail".

Examples: one light source, three colors only, silhouettes only, no faces shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class LimitRule(str, Enum):
    MONOCHROME = "monochrome"
    TRIADIC_ONLY = "triadic_only"
    SINGLE_LIGHT = "single_light"
    SILHOUETTE = "silhouette"
    NO_FACE = "no_face"
    CENTER_SUBJECT = "center_subject"
    HORIZON_LOW = "horizon_low"
    TEXTURE_ONLY = "texture_only"


@dataclass(frozen=True)
class ConstraintPack:
    rules: Tuple[LimitRule, ...]
    positive: str
    negative: str


_TRIGGERS: Tuple[Tuple[re.Pattern, LimitRule, str, str], ...] = (
    (
        re.compile(r"\b(monochrome only|black and white only|single hue)\b", re.I),
        LimitRule.MONOCHROME,
        "strict monochrome value design, no chroma distraction",
        "color accents, rainbow noise",
    ),
    (
        re.compile(r"\b(three colors? only|limited palette|3-color)\b", re.I),
        LimitRule.TRIADIC_ONLY,
        "only three hues plus neutrals, disciplined palette",
        "extra accent colors, palette drift",
    ),
    (
        re.compile(r"\b(one light|single light source|one lamp)\b", re.I),
        LimitRule.SINGLE_LIGHT,
        "exactly one motivated light source, coherent shadow fan",
        "multiple conflicting keys, flat ambient mush",
    ),
    (
        re.compile(r"\b(silhouette only|silhouettes?|backlit shape)\b", re.I),
        LimitRule.SILHOUETTE,
        "readable silhouette, interior detail suppressed, rim edge clarity",
        "interior detail in silhouette, gray fill",
    ),
    (
        re.compile(r"\b(faceless|no face|hidden face|turned away)\b", re.I),
        LimitRule.NO_FACE,
        "face obscured or turned away, identity through body language",
        "direct eye contact, visible facial features",
    ),
    (
        re.compile(r"\b(low horizon|horizon at bottom)\b", re.I),
        LimitRule.HORIZON_LOW,
        "sky-dominant composition, low horizon line, vast ceiling of atmosphere",
        "centered horizon, cramped sky",
    ),
    (
        re.compile(r"\b(texture study|macro texture only|surface abstract)\b", re.I),
        LimitRule.TEXTURE_ONLY,
        "abstract texture read, material truth, minimal figurative content",
        "busy scene narrative, random objects",
    ),
)


class CreativeConstraintEngine:
    def detect(self, prompt: str) -> List[LimitRule]:
        text = prompt or ""
        return [rule for pat, rule, _, _ in _TRIGGERS if pat.search(text)]

    def pack(self, prompt: str) -> ConstraintPack:
        rules = self.detect(prompt)
        if not rules:
            return ConstraintPack((), "", "")
        pos_parts: List[str] = []
        neg_parts: List[str] = []
        for pat, rule, pos, neg in _TRIGGERS:
            if rule in rules:
                pos_parts.append(pos)
                neg_parts.append(neg)
        return ConstraintPack(tuple(rules), ", ".join(pos_parts), ", ".join(neg_parts))

    def suggest_random(self, *, seed: int = 0) -> ConstraintPack:
        """Pick a creative limit when user wants surprise."""
        rules = list(LimitRule)
        r = rules[seed % len(rules)]
        for pat, rule, pos, neg in _TRIGGERS:
            if rule == r:
                return ConstraintPack((r,), pos, neg)
        return ConstraintPack((r,), "creative restriction discipline", "generic clutter")


__all__ = ["ConstraintPack", "CreativeConstraintEngine", "LimitRule"]
