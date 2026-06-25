"""
Cinema shot grammar — ECU, OTS, worm's eye, establishing, insert cut.

Film-director vocabulary; pairs with narrative/witness modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ShotType(str, Enum):
    ESTABLISHING = "establishing"
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE = "extreme_close"
    OVER_SHOULDER = "over_shoulder"
    POV = "pov"
    WORMS_EYE = "worms_eye"
    BIRDS_EYE = "birds_eye"
    INSERT = "insert"
    TWO_SHOT = "two_shot"


@dataclass(frozen=True)
class ShotPlan:
    shot: ShotType
    positive: str
    negative: str
    lens_hint: str


_SHOTS: Tuple[Tuple[re.Pattern, ShotType, str, str, str], ...] = (
    (
        re.compile(r"\b(establishing shot|extreme wide|city establishing)\b", re.I),
        ShotType.ESTABLISHING,
        "geography readable in one frame, story context, tiny human scale optional",
        "tight crop without context",
        "24mm environmental",
    ),
    (
        re.compile(r"\b(over the shoulder|OTS|over-shoulder)\b", re.I),
        ShotType.OVER_SHOULDER,
        "foreground shoulder soft blur, subject beyond, dialogue staging",
        "flat single plane, no depth staging",
        "50mm shallow",
    ),
    (
        re.compile(r"\b(POV|point of view|first person view)\b", re.I),
        ShotType.POV,
        "first-person hands optional, horizon at eye line, immersive",
        "third person floating camera",
        "wide POV lens",
    ),
    (
        re.compile(r"\b(worm'?s eye|from below|low angle hero)\b", re.I),
        ShotType.WORMS_EYE,
        "extreme low angle, towering subject, converging verticals",
        "eye-level default",
        "ultrawide low",
    ),
    (
        re.compile(r"\b(bird'?s eye|top down|aerial directly above)\b", re.I),
        ShotType.BIRDS_EYE,
        "plan view geometry, pattern read, map-like clarity",
        "tilted aerial confusion",
        "top-down orthographic feel",
    ),
    (
        re.compile(r"\b(insert shot|detail cut|macro insert)\b", re.I),
        ShotType.INSERT,
        "narrative object isolated, shallow depth, story prop clarity",
        "wide scene without insert focus",
        "macro insert",
    ),
    (
        re.compile(r"\b(two shot|two-shot|couple frame)\b", re.I),
        ShotType.TWO_SHOT,
        "two subjects balanced in frame, relationship staging",
        "orphan single subject when two intended",
        "50mm two-shot",
    ),
    (
        re.compile(r"\b(extreme close[- ]?up|ECU|eyes only)\b", re.I),
        ShotType.EXTREME_CLOSE,
        "skin pore scale, iris detail, crop at brow or lip",
        "medium shot distance",
        "100mm macro portrait",
    ),
)


class ShotGrammar:
    def plan(self, prompt: str) -> ShotPlan:
        text = prompt or ""
        for pat, shot, pos, neg, lens in _SHOTS:
            if pat.search(text):
                return ShotPlan(shot, pos, neg, lens)
        if re.search(r"\b(cinematic|film still|movie frame)\b", text, re.I):
            return ShotPlan(
                ShotType.MEDIUM,
                "cinematic framing, headroom discipline, lookroom toward action",
                "amateur snapshot framing",
                "35mm film still",
            )
        return ShotPlan(ShotType.MEDIUM, "", "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        pos = ", ".join(x for x in (p.positive, p.lens_hint) if x)
        return pos, p.negative


__all__ = ["ShotGrammar", "ShotPlan", "ShotType"]
