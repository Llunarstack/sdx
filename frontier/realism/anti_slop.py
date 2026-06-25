"""
Anti-AI-slop scanner — targets tells that make "photoreal" outputs feel synthetic.

Most models default to: plastic skin, over-sharpened eyes, HDR halos, symmetry drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class RealismTier(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    HYPERREAL = "hyperreal"
    DOCUMENTARY = "documentary"


@dataclass(frozen=True)
class AntiSlopPlan:
    tier: RealismTier
    positive: str
    negative: str
    microdetail_hint: str


_REALISM_TRIGGERS = re.compile(
    r"\b(photoreal|hyperreal|lifelike|dslr|raw photo|8k photo|ultra realistic|"
    r"real life|documentary photo|cinematic photo|portrait photo)\b",
    re.I,
)
_HYPER = re.compile(r"\b(hyperreal|8k|ultra realistic|extreme detail)\b", re.I)
_DOC = re.compile(r"\b(documentary|photojournalism|candid|street photo)\b", re.I)

_NEG_COMMON = (
    "ai generated look, plastic skin, waxy pores, over-smoothed skin, beauty filter, "
    "dead glassy eyes, oversharpened irises, symmetry perfection, uncanny valley, "
    "hdr halo, crunchy micro-contrast, ai soup texture, watermark, signature text"
)
_POS_STANDARD = (
    "natural camera response, subtle sensor noise, imperfect skin with pores and freckles, "
    "asymmetric facial micro-features, realistic highlight rolloff, true lens bokeh"
)
_POS_HYPER = (
    "micro-detail without crunch, subsurface skin scatter, individual hair strand variation, "
    "material-true specular, optical depth falloff, filmic tone mapping"
)
_POS_DOC = (
    "authentic candid moment, unposed gesture, environmental context truth, "
    "natural white balance drift, available light only"
)
_MICRO = "preserve pore-scale detail in focal plane only; avoid global oversharpen"


class AntiSlopScanner:
    def detect_tier(self, prompt: str) -> RealismTier:
        text = prompt or ""
        if not _REALISM_TRIGGERS.search(text):
            return RealismTier.NONE
        if _DOC.search(text):
            return RealismTier.DOCUMENTARY
        if _HYPER.search(text):
            return RealismTier.HYPERREAL
        return RealismTier.STANDARD

    def plan(self, prompt: str) -> AntiSlopPlan:
        tier = self.detect_tier(prompt)
        if tier == RealismTier.NONE:
            return AntiSlopPlan(tier, "", "", "")
        pos = {
            RealismTier.STANDARD: _POS_STANDARD,
            RealismTier.HYPERREAL: _POS_STANDARD + ", " + _POS_HYPER,
            RealismTier.DOCUMENTARY: _POS_DOC,
        }[tier]
        return AntiSlopPlan(tier, pos, _NEG_COMMON, _MICRO)

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["AntiSlopPlan", "AntiSlopScanner", "RealismTier"]
