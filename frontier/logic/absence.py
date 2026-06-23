"""
Model what must *not* occupy space — negative space as a first-class constraint.

Prompts like "empty bench", "no text", "clear sky" become explicit absence zones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class AbsenceConstraint:
    """Something the scene must lack or keep empty."""

    subject: str
    strength: float  # 0..1
    source_phrase: str
    negative_prompt_fragment: str


_ABSENCE_PATTERNS: Sequence[tuple[str, str, float]] = (
    (r"\bno\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.9),
    (r"\bwithout\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.85),
    (r"\bempty\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.75),
    (r"\bclear\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.7),
    (r"\bbare\s+([a-z][\w-]{1,24}s?)\b", r"\1", 0.65),
    (r"\bnegative\s+space\b", "clutter", 0.8),
    (r"\bminimal\s+background\b", "busy background, clutter", 0.7),
)


class AbsenceExtractor:
    """Pull absence constraints from natural-language prompts."""

    def extract(self, prompt: str) -> List[AbsenceConstraint]:
        text = (prompt or "").strip()
        if not text:
            return []

        out: List[AbsenceConstraint] = []
        seen: set[str] = set()
        for pattern, subject_tpl, strength in _ABSENCE_PATTERNS:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                phrase = m.group(0)
                if subject_tpl == r"\1" and m.lastindex:
                    subject = m.group(1).lower()
                else:
                    subject = subject_tpl
                if subject in seen:
                    continue
                seen.add(subject)
                neg = f"{subject}, {subject} everywhere, crowded {subject}"
                out.append(
                    AbsenceConstraint(
                        subject=subject,
                        strength=strength,
                        source_phrase=phrase,
                        negative_prompt_fragment=neg,
                    )
                )
        return out

    def merge_negative_prompt(self, base_negative: str, constraints: Sequence[AbsenceConstraint]) -> str:
        parts = [p.strip() for p in (base_negative or "").split(",") if p.strip()]
        for c in constraints:
            if c.strength >= 0.7:
                parts.append(c.negative_prompt_fragment)
        # dedupe preserving order
        seen: set[str] = set()
        merged: List[str] = []
        for p in parts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                merged.append(p)
        return ", ".join(merged)
