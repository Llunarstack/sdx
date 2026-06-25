"""Lens personality — vintage, anamorphic, tilt-shift, fisheye."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class LensPersonality(str, Enum):
    STANDARD = "standard"
    ANAMORPHIC = "anamorphic"
    FISHEYE = "fisheye"
    TILT_SHIFT = "tilt_shift"
    VINTAGE = "vintage"
    SOFT_PORTRAIT = "soft_portrait"
    MACRO = "macro"


@dataclass(frozen=True)
class LensCharacter:
    personality: LensPersonality
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, LensPersonality, str, str], ...] = (
    (
        re.compile(r"\b(anamorphic|cinemascope|oval bokeh|lens flare streak)\b", re.I),
        LensPersonality.ANAMORPHIC,
        "anamorphic oval bokeh, horizontal flare streaks, cinematic 2.39:1 feel",
        "circular bokeh on anamorphic scene, wrong aspect cues",
    ),
    (
        re.compile(r"\b(fisheye|ultra wide distortion|gopro)\b", re.I),
        LensPersonality.FISHEYE,
        "barrel distortion, curved horizon, immersive wide POV",
        "rectilinear straight lines in fisheye",
    ),
    (
        re.compile(r"\b(tilt[- ]shift|miniature effect|diorama lens)\b", re.I),
        LensPersonality.TILT_SHIFT,
        "tilt-shift plane of focus, miniature diorama effect, selective sharp band",
        "uniform sharpness with fake blur overlay",
    ),
    (
        re.compile(r"\b(vintage lens|film lens|helios|swirly bokeh|retro photo)\b", re.I),
        LensPersonality.VINTAGE,
        "vintage lens character, swirly bokeh, soft edge falloff, gentle flare",
        "clinical modern sharpness on vintage request",
    ),
    (
        re.compile(r"\b(dreamy portrait|soft focus|vintage glamour)\b", re.I),
        LensPersonality.SOFT_PORTRAIT,
        "soft portrait glow, flattering skin diffusion, gentle highlight bloom",
        "oversharpened pores, crunchy skin",
    ),
    (
        re.compile(r"\b(macro lens|extreme close-up|100mm macro)\b", re.I),
        LensPersonality.MACRO,
        "macro magnification, shallow DOF plane, diffraction softness at limit",
        "wide angle macro distortion",
    ),
)


class LensCharacterPlanner:
    def plan(self, prompt: str) -> LensCharacter:
        text = prompt or ""
        for pat, pers, pos, neg in _RULES:
            if pat.search(text):
                return LensCharacter(pers, pos, neg)
        return LensCharacter(LensPersonality.STANDARD, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        c = self.plan(prompt)
        return c.positive, c.negative


__all__ = ["LensCharacter", "LensCharacterPlanner", "LensPersonality"]
