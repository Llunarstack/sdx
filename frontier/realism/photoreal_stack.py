"""
Photoreal stack — lens, sensor, and lighting vocabulary for realism prompts.

Layers on anti-slop with camera-specific cues models often omit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from .anti_slop import AntiSlopScanner, RealismTier


@dataclass(frozen=True)
class PhotorealPlan:
    tier: RealismTier
    positive: str
    negative: str
    lens_hint: str
    sensor_hint: str


_LENS_PORTRAIT = re.compile(r"\b(portrait|85mm|50mm|headshot|bokeh)\b", re.I)
_LENS_WIDE = re.compile(r"\b(wide angle|24mm|16mm|landscape photo|architecture)\b", re.I)
_LENS_MACRO = re.compile(r"\b(macro|100mm macro|close-up|insect photo)\b", re.I)
_LENS_TELE = re.compile(r"\b(telephoto|200mm|sports|wildlife)\b", re.I)


class PhotorealStack:
    def __init__(self) -> None:
        self._slop = AntiSlopScanner()

    def plan(self, prompt: str) -> PhotorealPlan:
        slop = self._slop.plan(prompt)
        if slop.tier == RealismTier.NONE:
            return PhotorealPlan(RealismTier.NONE, "", "", "", "")

        lens = "natural perspective, mild lens vignette, subtle chromatic aberration at edges"
        sensor = "full-frame dynamic range, natural ISO grain in shadows"
        text = prompt or ""

        if _LENS_PORTRAIT.search(text):
            lens = "portrait lens compression, shallow depth of field, creamy bokeh discs"
        elif _LENS_WIDE.search(text):
            lens = "wide-angle perspective, controlled edge distortion, deep focus hyperfocal"
        elif _LENS_MACRO.search(text):
            lens = "macro shallow depth, diffraction softening at extreme close focus"
        elif _LENS_TELE.search(text):
            lens = "telephoto compression, subject isolation, motion-aware shutter"

        pos = ", ".join(p for p in (slop.positive, lens, sensor, slop.microdetail_hint) if p)
        neg = slop.negative
        return PhotorealPlan(slop.tier, pos, neg, lens, sensor)

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["PhotorealPlan", "PhotorealStack"]
