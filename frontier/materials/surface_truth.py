"""Surface material truth — metal, glass, fabric, skin, wood."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class SurfaceClass(str, Enum):
    METAL = "metal"
    GLASS = "glass"
    FABRIC = "fabric"
    SKIN = "skin"
    WOOD = "wood"
    STONE = "stone"
    LIQUID = "liquid"
    HAIR = "hair"


@dataclass(frozen=True)
class MaterialHint:
    surface: SurfaceClass
    positive: str
    negative: str


_SURFACES: Tuple[Tuple[re.Pattern, SurfaceClass, str, str], ...] = (
    (
        re.compile(r"\b(chrome|steel|brass|copper|metal|metallic|gold|silver)\b", re.I),
        SurfaceClass.METAL,
        "correct metal Fresnel, roughness breakup, specular highlight rolloff",
        "plastic painted as metal, mirror chrome everywhere",
    ),
    (
        re.compile(r"\b(glass|window|lens|crystal|transparent|refraction)\b", re.I),
        SurfaceClass.GLASS,
        "refraction and caustics, clean transmission, edge darkening",
        "sticker transparency, no refraction",
    ),
    (
        re.compile(r"\b(silk|velvet|linen|cotton|fabric|cloth|dress|suit)\b", re.I),
        SurfaceClass.FABRIC,
        "weave micro-texture, fold tension, subsurface on thin edges",
        "plastic wrap cloth, stiff cardboard folds",
    ),
    (
        re.compile(r"\b(skin|face|portrait|hands|body)\b", re.I),
        SurfaceClass.SKIN,
        "subsurface scatter, pore micro-detail in focal zone, natural oil sheen",
        "plastic wax skin, uniform pore stamp",
    ),
    (
        re.compile(r"\b(wood|oak|pine|mahogany|grain|timber)\b", re.I),
        SurfaceClass.WOOD,
        "directional wood grain, anisotropic highlight, edge wear",
        "repeating texture tile, plastic wood",
    ),
    (
        re.compile(r"\b(stone|marble|granite|concrete|brick)\b", re.I),
        SurfaceClass.STONE,
        "mineral grain, weathering variation, contact shadow grounding",
        "uniform noise rock, floating stones",
    ),
    (
        re.compile(r"\b(water|wine|liquid|pouring|splash|ocean)\b", re.I),
        SurfaceClass.LIQUID,
        "surface tension, caustics, refraction, believable viscosity",
        "solid blue plastic water",
    ),
    (
        re.compile(r"\b(hair|braid|ponytail|fur|mane)\b", re.I),
        SurfaceClass.HAIR,
        "individual strand groups, specular along fiber, flyaway variation",
        "solid helmet hair, plastic fur clumps",
    ),
)


class MaterialPlanner:
    def scan(self, prompt: str) -> List[MaterialHint]:
        text = prompt or ""
        hits: List[MaterialHint] = []
        seen: set[SurfaceClass] = set()
        for pat, surf, pos, neg in _SURFACES:
            if pat.search(text) and surf not in seen:
                hits.append(MaterialHint(surf, pos, neg))
                seen.add(surf)
        return hits

    def fragments(self, prompt: str, max_surfaces: int = 4) -> Tuple[str, str]:
        hints = self.scan(prompt)[:max_surfaces]
        if not hints:
            return "", ""
        pos = ", ".join(h.positive for h in hints)
        neg = ", ".join(h.negative for h in hints)
        return pos, neg


__all__ = ["MaterialHint", "MaterialPlanner", "SurfaceClass"]
