"""
Tiered content policy — pre-generation checks for prohibited solicitations.

Does not replace legal counsel or post-gen classifiers. Blocks high-confidence
sexual+minor prompt combinations; steers adult NSFW toward adult-coded language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class PolicyTier(str, Enum):
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    STEER = "steer"
    REFUSE = "refuse"


@dataclass(frozen=True)
class SafetyReport:
    decision: PolicyDecision
    tier: PolicyTier
    reasons: Tuple[str, ...]
    steer_positive: str = ""
    steer_negative: str = ""


_SEXUAL = re.compile(
    r"\b(nude|naked|nsfw|explicit|erotic|sex|porn|hentai|lewd|boudoir|lingerie|"
    r"topless|genital|xxx|ecchi)\b",
    re.I,
)
_YOUTH = re.compile(
    r"\b(child|children|kid|kids|minor|underage|loli|shota|preteen|pedo|"
    r"young girl|young boy|schoolgirl|schoolboy|teen girl|teen boy|"
    r"elementary|middle school|high school student)\b",
    re.I,
)
_YOUTH_EUPHEM = re.compile(r"\b(barely legal|loli|shota|petite girl|tiny girl)\b", re.I)
_ADULT_AFFIRM = re.compile(
    r"\b(adult woman|adult man|mature woman|mature man|25[- ]year|30[- ]year|"
    r"grown woman|grown man|of age)\b",
    re.I,
)

_STEER_POS = "adult subject with mature facial features, adult proportions, clearly of age"
_STEER_NEG = "child, minor, underage, young looking, loli, shota, school uniform fetish"


class ContentPolicy:
    def __init__(self, tier: PolicyTier = PolicyTier.MODERATE) -> None:
        self.tier = tier

    def evaluate(self, prompt: str, *, negative: str = "") -> SafetyReport:
        if self.tier == PolicyTier.OFF:
            return SafetyReport(PolicyDecision.ALLOW, self.tier, ())

        text = f"{prompt} {negative}".strip()
        reasons: List[str] = []

        sexual = bool(_SEXUAL.search(text))
        youth = bool(_YOUTH.search(text))
        euph = bool(_YOUTH_EUPHEM.search(text))
        adult_ok = bool(_ADULT_AFFIRM.search(text))

        if youth or euph:
            reasons.append("youth_or_euphemism_detected")
        if sexual:
            reasons.append("sexual_content_detected")

        if self.tier == PolicyTier.STRICT and (youth or euph):
            return SafetyReport(PolicyDecision.REFUSE, self.tier, tuple(reasons))

        if sexual and (youth or euph):
            if self.tier == PolicyTier.MODERATE and not euph and adult_ok:
                return SafetyReport(
                    PolicyDecision.STEER,
                    self.tier,
                    tuple(reasons + ["adult_affirmation_present"]),
                    steer_positive=_STEER_POS,
                    steer_negative=_STEER_NEG,
                )
            return SafetyReport(PolicyDecision.REFUSE, self.tier, tuple(reasons))

        if sexual and not adult_ok and self.tier == PolicyTier.MODERATE:
            return SafetyReport(
                PolicyDecision.STEER,
                self.tier,
                ("adult_steering_recommended",),
                steer_positive=_STEER_POS,
                steer_negative=_STEER_NEG,
            )

        return SafetyReport(PolicyDecision.ALLOW, self.tier, tuple(reasons))

    def apply_steering(self, prompt: str, report: SafetyReport) -> Tuple[str, str]:
        if report.decision != PolicyDecision.STEER:
            return prompt, ""
        pos_extra = report.steer_positive
        neg_extra = report.steer_negative
        new_prompt = f"{prompt}, {pos_extra}" if prompt and pos_extra else prompt or pos_extra
        return new_prompt, neg_extra


__all__ = ["ContentPolicy", "PolicyDecision", "PolicyTier", "SafetyReport"]
