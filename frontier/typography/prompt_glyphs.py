"""Typography prompt packs for legible in-image text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TypographyPlan:
    quoted_text: Tuple[str, ...]
    positive: str
    negative: str
    cfg_boost: float


_QUOTED = re.compile(r'"([^"]{1,80})"')
_BRACKET = re.compile(r"\[text:\s*([^\]]{1,80})\]", re.I)
_SIGN = re.compile(r"\b(sign|logo|poster|banner|headline|typography|lettering|title card)\b", re.I)


class TypographyPlanner:
    def extract_quotes(self, prompt: str) -> List[str]:
        text = prompt or ""
        out: List[str] = []
        out.extend(_QUOTED.findall(text))
        out.extend(_BRACKET.findall(text))
        return list(dict.fromkeys(t.strip() for t in out if t.strip()))

    def plan(self, prompt: str) -> TypographyPlan:
        quotes = tuple(self.extract_quotes(prompt))
        has_typo = bool(quotes) or bool(_SIGN.search(prompt or ""))
        if not has_typo:
            return TypographyPlan((), "", "", 0.0)

        spell = ""
        if quotes:
            spell = ", ".join(f'legible text reading exactly "{q}"' for q in quotes[:3])

        pos = ", ".join(
            p
            for p in (
                spell,
                "crisp typography, correct spelling, high contrast text-background separation",
                "professional kerning and baseline alignment",
            )
            if p
        )
        neg = (
            "misspelled text, garbled letters, melted typography, nonsense characters, "
            "duplicate letters, watermark, signature overlay"
        )
        return TypographyPlan(quotes, pos, neg, cfg_boost=1.15 if quotes else 1.08)

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["TypographyPlan", "TypographyPlanner"]
