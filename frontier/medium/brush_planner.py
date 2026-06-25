"""
Brush-stroke planner — direction, pressure, and edge behavior per medium.

Complements ``config.defaults.art_mediums`` with stroke-level vocabulary models understand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class StrokeStyle(str, Enum):
    IMPASTO = "impasto"
    WET_ON_WET = "wet_on_wet"
    DRY_BRUSH = "dry_brush"
    HATCHING = "hatching"
    STIPPLING = "stippling"
    SPRAY = "spray"
    PALETTE_KNIFE = "palette_knife"
    CALLIGRAPHIC = "calligraphic"
    FLAT_FILL = "flat_fill"
    SOFT_BLEND = "soft_blend"


@dataclass(frozen=True)
class BrushPlan:
    primary_style: StrokeStyle
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, StrokeStyle, str, str], ...] = (
    (
        re.compile(r"\b(impasto|thick paint|palette knife|alla prima)\b", re.I),
        StrokeStyle.IMPASTO,
        "visible impasto ridges, directional brush load, thick paint peaks catching light",
        "flat digital gradient, airbrush mush",
    ),
    (
        re.compile(r"\b(watercolor|wet[- ]on[- ]wet|wash)\b", re.I),
        StrokeStyle.WET_ON_WET,
        "pigment bloom on paper tooth, controlled backrun edges, transparent glaze layers",
        "opaque plastic fill, random stain noise",
    ),
    (
        re.compile(r"\b(dry brush|scrubbed brush|broken stroke)\b", re.I),
        StrokeStyle.DRY_BRUSH,
        "broken bristle drag marks, canvas weave catching through, scratchy edge energy",
        "uniform smooth blend, textureless stroke",
    ),
    (
        re.compile(r"\b(crosshatch|hatching|engraving|etching|woodcut|linocut)\b", re.I),
        StrokeStyle.HATCHING,
        "consistent hatch angle families, line weight hierarchy, plate pressure variation",
        "random scribble noise, wobbly parallel lines",
    ),
    (
        re.compile(r"\b(stipple|pointillism|dotwork)\b", re.I),
        StrokeStyle.STIPPLING,
        "rhythmic dot clusters, optical color mixing, disciplined dot scale",
        "uniform noise spray, muddy dot soup",
    ),
    (
        re.compile(r"\b(spray paint|street art|graffiti|aerosol)\b", re.I),
        StrokeStyle.SPRAY,
        "soft overspray falloff, cap pressure variation, layered stencil edges",
        "gaussian blur smear, incoherent drip physics",
    ),
    (
        re.compile(r"\b(calligraphy|brush lettering|sumi)\b", re.I),
        StrokeStyle.CALLIGRAPHIC,
        "pressure-thick to thin stroke taper, intentional ink pooling at turns",
        "uniform vector stroke, wobbly sans-serif",
    ),
    (
        re.compile(r"\b(cel shading|flat color|vector)\b", re.I),
        StrokeStyle.FLAT_FILL,
        "decisive flat fills, clean shape breaks, minimal accidental gradient",
        "unintentional photoreal gradient leak",
    ),
    (
        re.compile(r"\b(airbrush|soft blend|painterly blend)\b", re.I),
        StrokeStyle.SOFT_BLEND,
        "controlled soft transitions, edge hierarchy preserved, no muddy midtone soup",
        "banded gradient, plastic skin airbrush",
    ),
)


class BrushPlanner:
    def plan(self, prompt: str) -> BrushPlan | None:
        text = prompt or ""
        for pat, style, pos, neg in _RULES:
            if pat.search(text):
                return BrushPlan(primary_style=style, positive=pos, negative=neg)
        if re.search(r"\b(oil|acrylic|gouache|pastel|charcoal|ink)\b", text, re.I):
            return BrushPlan(
                StrokeStyle.SOFT_BLEND,
                "intentional hand-made stroke variation, edge variety, medium-true surface response",
                "ai soup smear, texture spam, over-blended plastic",
            )
        return None

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        if p is None:
            return "", ""
        return p.positive, p.negative


__all__ = ["BrushPlan", "BrushPlanner", "StrokeStyle"]
