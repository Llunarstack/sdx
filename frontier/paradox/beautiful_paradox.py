"""
Beautiful paradox — Escher, infinite loops, day-night in one frame.

Pairs with ``logic/contradiction``: if paradox is *requested*, suppress auto-resolve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class BeautifulParadox(str, Enum):
    ESCHER = "escher"
    INFINITE_LOOP = "infinite_loop"
    DUAL_TIME = "dual_time"
    INSIDE_OUTSIDE = "inside_outside"
    NONE = "none"


@dataclass(frozen=True)
class ParadoxPlan:
    kind: BeautifulParadox
    positive: str
    negative: str
    suppress_contradiction_resolve: bool


_PATTERNS: Tuple[Tuple[re.Pattern, BeautifulParadox, str, str], ...] = (
    (
        re.compile(r"\b(escher|impossible stairs|penrose|relativity)\b", re.I),
        BeautifulParadox.ESCHER,
        "consistent impossible geometry, paradox architecture with single light source",
        "broken perspective without internal logic",
    ),
    (
        re.compile(r"\b(infinite loop|ouroboros|perpetual waterfall|droste)\b", re.I),
        BeautifulParadox.INFINITE_LOOP,
        "recursive visual loop, seamless self-reference, hypnotic repetition",
        "obvious seam in loop, broken recursion",
    ),
    (
        re.compile(r"\b(day and night|sun and moon same sky|dusk and dawn)\b", re.I),
        BeautifulParadox.DUAL_TIME,
        "split-time sky, coherent horizon, two times in one frame by design",
        "muddy gray without time read",
    ),
    (
        re.compile(r"\b(inside outside|interior landscape|room with sky)\b", re.I),
        BeautifulParadox.INSIDE_OUTSIDE,
        "interior-exterior boundary dissolve, architectural paradox",
        "random sky pasted in room",
    ),
)


class ParadoxKeeper:
    def plan(self, prompt: str) -> ParadoxPlan:
        text = prompt or ""
        for pat, kind, pos, neg in _PATTERNS:
            if pat.search(text):
                return ParadoxPlan(kind, pos, neg, suppress_contradiction_resolve=True)
        return ParadoxPlan(BeautifulParadox.NONE, "", "", False)

    def fragments(self, prompt: str) -> Tuple[str, str, bool]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.suppress_contradiction_resolve


__all__ = ["BeautifulParadox", "ParadoxKeeper", "ParadoxPlan"]
