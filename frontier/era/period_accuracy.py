"""Historical era accuracy — costume, architecture, technology cues."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class HistoricalEra(str, Enum):
    ANCIENT = "ancient"
    MEDIEVAL = "medieval"
    RENAISSANCE = "renaissance"
    VICTORIAN = "victorian"
    EDWARDIAN = "edwardian"
    ART_DECO = "art_deco"
    MID_CENTURY = "mid_century"
    EIGHTIES = "eighties"
    NINETIES = "nineties"
    NEAR_FUTURE = "near_future"
    GENERIC = "generic"


@dataclass(frozen=True)
class EraPlan:
    era: HistoricalEra
    positive: str
    negative: str


_ERAS: Tuple[Tuple[re.Pattern, HistoricalEra, str, str], ...] = (
    (
        re.compile(r"\b(ancient rome|greek|egyptian|roman|spartan)\b", re.I),
        HistoricalEra.ANCIENT,
        "period-accurate drapery and armor, pre-industrial materials, patina",
        "modern zippers, wristwatches, synthetic fabric",
    ),
    (
        re.compile(r"\b(medieval|knight|castle|middle ages|viking)\b", re.I),
        HistoricalEra.MEDIEVAL,
        "medieval garment construction, hand-forged metal, timber and stone architecture",
        "modern eyeglasses, printed text, machine stitching",
    ),
    (
        re.compile(r"\b(renaissance|15th century|16th century|leonardo|medici)\b", re.I),
        HistoricalEra.RENAISSANCE,
        "renaissance textile and silhouette, oil-lamp lighting, classical proportion",
        "neon signs, plastic buttons, contemporary haircut",
    ),
    (
        re.compile(r"\b(victorian|19th century|steampunk|industrial revolution)\b", re.I),
        HistoricalEra.VICTORIAN,
        "victorian tailoring, gaslight warmth, brass and iron technology",
        "smartphones, LED lights, modern sneakers",
    ),
    (
        re.compile(r"\b(art deco|1920s|1930s|gatsby|flapper)\b", re.I),
        HistoricalEra.ART_DECO,
        "art deco geometry, jazz age fashion, geometric typography cues",
        "modern minimal UI, contemporary athleisure",
    ),
    (
        re.compile(r"\b(1950s|1960s|mid century|mid-century|mad men)\b", re.I),
        HistoricalEra.MID_CENTURY,
        "mid-century furniture lines, period hair and makeup, analog technology",
        "flat screen TVs, modern logos, contemporary streetwear",
    ),
    (
        re.compile(r"\b(1980s|80s|synthwave|retro 80)\b", re.I),
        HistoricalEra.EIGHTIES,
        "1980s fashion volume, analog grain optional, period correct electronics",
        "smartphones, 2020s streetwear, modern EV cars",
    ),
    (
        re.compile(r"\b(1990s|90s|grunge|y2k)\b", re.I),
        HistoricalEra.NINETIES,
        "1990s silhouette and tech, CRT optional, period denim and outerwear",
        "modern slim-fit only, 2020s phone design",
    ),
    (
        re.compile(r"\b(near future|2050|cyberpunk|sci-fi city)\b", re.I),
        HistoricalEra.NEAR_FUTURE,
        "coherent future design language, worn technology, plausible infrastructure",
        "random hologram spam, inconsistent tech eras mixed",
    ),
)


class EraPlanner:
    def plan(self, prompt: str) -> EraPlan:
        text = prompt or ""
        for pat, era, pos, neg in _ERAS:
            if pat.search(text):
                return EraPlan(era, pos, neg)
        return EraPlan(HistoricalEra.GENERIC, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["EraPlan", "EraPlanner", "HistoricalEra"]
