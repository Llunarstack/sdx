"""
Uncertainty-aware generation — escalate compute when the prompt is ambiguous.

Few image products expose "I'm not sure what you mean" before burning GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class UncertaintySignal(str, Enum):
    VAGUE_QUANTITY = "vague_quantity"
    CONFLICTING_STYLE = "conflicting_style"
    UNGROUNDED_REFERENCE = "ungrounded_reference"
    ORPHAN_PRONOUN = "orphan_pronoun"
    OVERLOADED_SCENE = "overloaded_scene"


@dataclass(frozen=True)
class UncertaintyReport:
    score: float  # 0..1
    signals: tuple[UncertaintySignal, ...]
    clarification_questions: tuple[str, ...]
    cfg_boost: float = 0.0
    recommend_best_of_n: int = 1


class ConfidenceGate:
    """Heuristic uncertainty before sampling."""

    _VAGUE = re.compile(r"\bsome\b|\bfew\b|\bseveral\b|\bstuff\b|\bthings\b", re.I)
    _PRONOUN = re.compile(r"\b(it|they|them|that one|the other)\b", re.I)
    _STYLE_MIX = re.compile(
        r"(photoreal|hyperreal|3d render).*(cartoon|anime|illustration)|"
        r"(cartoon|anime).*(photoreal|hyperreal)",
        re.I | re.S,
    )

    def analyze(self, prompt: str, *, contradiction_count: int = 0) -> UncertaintyReport:
        text = (prompt or "").strip()
        signals: List[UncertaintySignal] = []
        questions: List[str] = []

        if self._VAGUE.search(text):
            signals.append(UncertaintySignal.VAGUE_QUANTITY)
            questions.append("How many subjects, and how large in frame?")

        if self._PRONOUN.search(text) and len(text.split()) < 25:
            signals.append(UncertaintySignal.ORPHAN_PRONOUN)
            questions.append("What does 'it/they' refer to?")

        if self._STYLE_MIX.search(text):
            signals.append(UncertaintySignal.CONFLICTING_STYLE)
            questions.append("Pick one render style: photo, illustration, or 3D?")

        clauses = [c.strip() for c in text.split(",") if c.strip()]
        if len(clauses) >= 7:
            signals.append(UncertaintySignal.OVERLOADED_SCENE)
            questions.append("Which 2–3 elements matter most?")

        if contradiction_count > 0:
            signals.append(UncertaintySignal.CONFLICTING_STYLE)
            questions.append("Resolve contradictory lighting/time/space cues.")

        score = min(1.0, 0.12 * len(signals) + 0.08 * contradiction_count)
        cfg_boost = 0.15 * score
        bon = 1 if score < 0.35 else (2 if score < 0.6 else 4)

        return UncertaintyReport(
            score=score,
            signals=tuple(signals),
            clarification_questions=tuple(questions[:3]),
            cfg_boost=cfg_boost,
            recommend_best_of_n=bon,
        )


__all__ = ["ConfidenceGate", "UncertaintyReport", "UncertaintySignal"]
