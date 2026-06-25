"""
Mature-content quality guidance — anatomical and lighting coherence for adult art.

This module improves *render quality* for artistic nude, boudoir, and explicit prompts.
It does not block or filter prompts; pair with your own safety policy if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class MatureClass(str, Enum):
    NONE = "none"
    ARTISTIC_NUDE = "artistic_nude"
    BOUDOIR = "boudoir"
    EXPLICIT = "explicit"
    PINUP = "pinup"


@dataclass(frozen=True)
class MaturePlan:
    content_class: MatureClass
    positive: str
    negative: str
    lighting_hint: str
    anatomy_mode: str  # lite | strong — forwarded to body planner


_NUDE = re.compile(
    r"\b(nude|naked|topless|artistic nude|figure study|life drawing|nsfw)\b",
    re.I,
)
_BOUDOIR = re.compile(r"\b(boudoir|lingerie|sensual|intimate portrait|bedroom)\b", re.I)
_EXPLICIT = re.compile(
    r"\b(explicit|erotic|sex|intercourse|genital|xxx|porn|hentai|ecchi|lewd)\b",
    re.I,
)
_PINUP = re.compile(r"\b(pin[- ]?up|glamour|playboy style|cheesecake)\b", re.I)

_POS_BASE = (
    "natural adult proportions, believable skin texture with subsurface scatter, "
    "coherent anatomy without plastic smoothing, respectful lighting that models form"
)
_NEG_BASE = (
    "plastic doll skin, barbie anatomy, impossible proportions, airbrushed mannequin, "
    "wax figure, asymmetric nipples placement error, floating limbs, censored blur patch"
)
_LIGHT_SOFT = "soft window key with gentle fill, form-revealing but not flat frontal flash"
_LIGHT_DRAMATIC = "motivated rim and bounce light, cinematic falloff, preserved highlight rolloff"
_LIGHT_PINUP = "classic glamour key with controlled specular on skin, clean background separation"


class MatureGuidance:
    def classify(self, prompt: str) -> MatureClass:
        text = prompt or ""
        if _EXPLICIT.search(text):
            return MatureClass.EXPLICIT
        if _BOUDOIR.search(text):
            return MatureClass.BOUDOIR
        if _PINUP.search(text):
            return MatureClass.PINUP
        if _NUDE.search(text):
            return MatureClass.ARTISTIC_NUDE
        return MatureClass.NONE

    def plan(self, prompt: str) -> MaturePlan:
        cls = self.classify(prompt)
        if cls == MatureClass.NONE:
            return MaturePlan(cls, "", "", "", "none")

        anatomy = "strong" if cls in (MatureClass.EXPLICIT, MatureClass.ARTISTIC_NUDE) else "lite"
        lighting = {
            MatureClass.ARTISTIC_NUDE: _LIGHT_SOFT,
            MatureClass.BOUDOIR: _LIGHT_DRAMATIC,
            MatureClass.EXPLICIT: _LIGHT_DRAMATIC,
            MatureClass.PINUP: _LIGHT_PINUP,
        }[cls]

        extra_pos = ""
        extra_neg = "child, minor, underage, young looking, loli, shota"
        if cls == MatureClass.PINUP:
            extra_pos = "classic glamour pose readability, confident silhouette"
        if cls == MatureClass.BOUDOIR:
            extra_pos = "intimate but tasteful composition, fabric interaction truth"

        pos = ", ".join(p for p in (_POS_BASE, extra_pos, lighting) if p)
        neg = ", ".join(p for p in (_NEG_BASE, extra_neg) if p)
        return MaturePlan(cls, pos, neg, lighting, anatomy)

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["MatureClass", "MatureGuidance", "MaturePlan"]
