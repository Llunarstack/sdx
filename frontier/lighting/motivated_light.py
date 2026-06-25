"""Motivated light setups — Rembrandt, butterfly, split, rim, etc."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class LightPattern(str, Enum):
    REMBRANDT = "rembrandt"
    BUTTERFLY = "butterfly"
    SPLIT = "split"
    RIM = "rim"
    SILHOUETTE = "silhouette"
    OVERCAST = "overcast"
    GOLDEN_HOUR = "golden_hour"
    NEON = "neon"


@dataclass(frozen=True)
class LightSetup:
    pattern: LightPattern
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, LightPattern, str, str], ...] = (
    (
        re.compile(r"\b(rembrandt|triangle shadow|classic portrait light)\b", re.I),
        LightPattern.REMBRANDT,
        "Rembrandt lighting, triangle highlight on shadow cheek, soft key from 45 degrees",
        "flat on-camera flash, no shadow modeling",
    ),
    (
        re.compile(r"\b(butterfly|paramount|glamour light|beauty dish)\b", re.I),
        LightPattern.BUTTERFLY,
        "butterfly lighting, shadow under nose, glamour key above lens axis",
        "harsh under-chin shadow without fill, raccoon eyes",
    ),
    (
        re.compile(r"\b(split light|half face shadow|noir)\b", re.I),
        LightPattern.SPLIT,
        "split lighting, half face in shadow, dramatic contrast ratio",
        "muddy flat gray on both halves",
    ),
    (
        re.compile(r"\b(rim light|backlight|edge light|halo)\b", re.I),
        LightPattern.RIM,
        "strong rim separation from background, edge highlight on hair and shoulders",
        "subject melts into background, no edge separation",
    ),
    (
        re.compile(r"\b(silhouette|backlit figure| contre-jour)\b", re.I),
        LightPattern.SILHOUETTE,
        "silhouette against bright background, readable contour shape",
        "filled shadow with random detail, gray silhouette",
    ),
    (
        re.compile(r"\b(golden hour|sunset light|magic hour)\b", re.I),
        LightPattern.GOLDEN_HOUR,
        "warm golden hour key, long shadows, amber sky bounce fill",
        "neutral noon white balance on sunset scene",
    ),
    (
        re.compile(r"\b(neon|cyberpunk|rgb lights|club lighting)\b", re.I),
        LightPattern.NEON,
        "motivated neon color separation, magenta-cyan edge lights, wet surface reflections",
        "random rainbow without source, neon soup",
    ),
    (
        re.compile(r"\b(overcast|soft daylight|cloudy day)\b", re.I),
        LightPattern.OVERCAST,
        "soft omnidirectional skylight, low contrast, even skin modeling",
        "hard sun shadow on overcast day",
    ),
)


class LightingPlanner:
    def plan(self, prompt: str) -> LightSetup:
        text = prompt or ""
        for pat, pattern, pos, neg in _RULES:
            if pat.search(text):
                return LightSetup(pattern, pos, neg)
        if re.search(r"\b(portrait|studio|cinematic)\b", text, re.I):
            return LightSetup(
                LightPattern.REMBRANDT,
                "motivated key light, coherent shadow direction, fill ratio control",
                "conflicting light directions, floating shadows",
            )
        return LightSetup(LightPattern.OVERCAST, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        s = self.plan(prompt)
        return s.positive, s.negative


__all__ = ["LightPattern", "LightSetup", "LightingPlanner"]
