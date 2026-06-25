"""DOF and focus intent — isolation, discovery, overwhelm."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class FocusIntent(str, Enum):
    ISOLATE = "isolate"
    DISCOVER = "discover"
    OVERWHELM = "overwhelm"
    LAYERED = "layered"
    NONE = "none"


@dataclass(frozen=True)
class FocalStory:
    intent: FocusIntent
    positive: str
    negative: str
    cfg_hint: float


_RULES: Tuple[Tuple[re.Pattern, FocusIntent, str, str, float], ...] = (
    (
        re.compile(r"\b(shallow dof|bokeh|blurred background|subject isolation)\b", re.I),
        FocusIntent.ISOLATE,
        "shallow depth isolates subject emotionally, creamy bokeh separation",
        "subject and background equally sharp",
        1.05,
    ),
    (
        re.compile(r"\b(rack focus|focus pull|revealing detail)\b", re.I),
        FocusIntent.DISCOVER,
        "foreground blur yields to sharp discovery plane, narrative reveal",
        "uniform focus everywhere",
        1.08,
    ),
    (
        re.compile(r"\b(everything sharp|deep focus|hyperfocal|landscape clarity)\b", re.I),
        FocusIntent.OVERWHELM,
        "deep focus overwhelming detail, environmental complexity readable",
        "fake tilt-shift on wide scene",
        1.0,
    ),
    (
        re.compile(r"\b(foreground blur|layers of depth|through the reeds)\b", re.I),
        FocusIntent.LAYERED,
        "multiple depth layers, peek-through framing, voyeur discovery",
        "single plane cardboard",
        1.06,
    ),
)


class FocalStoryteller:
    def plan(self, prompt: str) -> FocalStory:
        text = prompt or ""
        for pat, intent, pos, neg, cfg in _RULES:
            if pat.search(text):
                return FocalStory(intent, pos, neg, cfg)
        return FocalStory(FocusIntent.NONE, "", "", 1.0)

    def fragments(self, prompt: str) -> Tuple[str, str, float]:
        p = self.plan(prompt)
        return p.positive, p.negative, p.cfg_hint


__all__ = ["FocalStory", "FocalStoryteller", "FocusIntent"]
