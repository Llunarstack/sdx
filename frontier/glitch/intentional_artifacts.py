"""Glitch as aesthetic choice — VHS, datamosh, scanlines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class GlitchAesthetic(str, Enum):
    VHS = "vhs"
    DATAMOSH = "datamosh"
    SCANLINE = "scanline"
    RGB_SPLIT = "rgb_split"
    PIXEL_SORT = "pixel_sort"
    NONE = "none"


@dataclass(frozen=True)
class GlitchPlan:
    aesthetic: GlitchAesthetic
    positive: str
    negative: str
    serendipity_boost: float


_RULES: Tuple[Tuple[re.Pattern, GlitchAesthetic, str, str, float], ...] = (
    (
        re.compile(r"\b(VHS|tape warp|tracking error|analog video)\b", re.I),
        GlitchAesthetic.VHS,
        "VHS head-switch noise, chroma bleed, tape curvature",
        "clean digital 4k on VHS request",
        0.1,
    ),
    (
        re.compile(r"\b(datamosh|compression artifact|mpeg glitch)\b", re.I),
        GlitchAesthetic.DATAMOSH,
        "intentional macroblock smear, motion-coded glitch bands",
        "random jpeg blocks",
        0.15,
    ),
    (
        re.compile(r"\b(scanlines?|CRT|tube TV)\b", re.I),
        GlitchAesthetic.SCANLINE,
        "CRT aperture grille, phosphor glow, barrel edge",
        "flat LCD on CRT request",
        0.05,
    ),
    (
        re.compile(r"\b(chromatic aberration|RGB split|glitch offset)\b", re.I),
        GlitchAesthetic.RGB_SPLIT,
        "channel offset at high contrast edges, prismatic fringe",
        "uniform blur",
        0.08,
    ),
    (
        re.compile(r"\b(pixel sort|sorted pixels|glitch art)\b", re.I),
        GlitchAesthetic.PIXEL_SORT,
        "directional pixel sort streaks along luminance edges",
        "random salt noise",
        0.12,
    ),
)


class GlitchPlanner:
    def plan(self, prompt: str) -> GlitchPlan:
        text = prompt or ""
        for pat, aes, pos, neg, boost in _RULES:
            if pat.search(text):
                return GlitchPlan(aes, pos, neg, boost)
        return GlitchPlan(GlitchAesthetic.NONE, "", "", 0.0)

    def fragments(self, prompt: str) -> Tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.serendipity_boost


__all__ = ["GlitchAesthetic", "GlitchPlan", "GlitchPlanner"]
