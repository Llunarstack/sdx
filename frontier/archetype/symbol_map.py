"""Jungian-adjacent symbols as compositional anchors — artistic, not preachy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ArchetypeSymbol(str, Enum):
    THRESHOLD = "threshold"
    MIRROR = "mirror"
    LABYRINTH = "labyrinth"
    CHALICE = "chalice"
    MASK = "mask"
    BRIDGE = "bridge"
    NONE = "none"


@dataclass(frozen=True)
class ArchetypePlan:
    symbol: ArchetypeSymbol
    positive: str
    negative: str


_MAP: Tuple[Tuple[re.Pattern, ArchetypeSymbol, str, str], ...] = (
    (
        re.compile(r"\b(threshold|doorway|gate|portal|liminal)\b", re.I),
        ArchetypeSymbol.THRESHOLD,
        "liminal threshold framing, passage between worlds, light spill from beyond",
        "random door pasted, no spatial logic",
    ),
    (
        re.compile(r"\b(mirror|reflection|doppelganger|twin)\b", re.I),
        ArchetypeSymbol.MIRROR,
        "meaningful mirror symmetry, identity doubling, reflective narrative",
        "broken mirror noise, accidental clone face",
    ),
    (
        re.compile(r"\b(labyrinth|maze|minotaur|lost paths)\b", re.I),
        ArchetypeSymbol.LABYRINTH,
        "recursive path geometry, center-oriented composition, maze readability",
        "random wall maze, no exit logic",
    ),
    (
        re.compile(r"\b(chalice|grail|offering|ritual cup)\b", re.I),
        ArchetypeSymbol.CHALICE,
        "symbolic vessel focal point, ceremonial light, reverent scale",
        "generic cup without symbolic weight",
    ),
    (
        re.compile(r"\b(mask|masked|masquerade|hidden identity)\b", re.I),
        ArchetypeSymbol.MASK,
        "mask as identity boundary, eye holes readable, material truth",
        "face blur pretending to be mask",
    ),
    (
        re.compile(r"\b(bridge|crossing|span|connecting shores)\b", re.I),
        ArchetypeSymbol.BRIDGE,
        "bridge as narrative connector, perspective along span, destination implied",
        "floating bridge without support",
    ),
)


class SymbolMapEngine:
    def plan(self, prompt: str) -> ArchetypePlan:
        text = prompt or ""
        for pat, sym, pos, neg in _MAP:
            if pat.search(text):
                return ArchetypePlan(sym, pos, neg)
        return ArchetypePlan(ArchetypeSymbol.NONE, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["ArchetypePlan", "ArchetypeSymbol", "SymbolMapEngine"]
