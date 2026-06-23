"""
Single-frame temporal implication: the image is a *slice* of a longer event.

Detects "about to fall", "just landed", "moment before" → sampling hints (early vs late steps).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class MomentPhase(str, Enum):
    STATIC = "static"
    ANTICIPATION = "anticipation"  # before the action
    CLIMAX = "climax"  # peak action
    AFTERMATH = "aftermath"  # just after


@dataclass(frozen=True)
class MomentCue:
    phase: MomentPhase
    confidence: float
    verb: str
    prompt_fragments: List[str]
    """Per-step emphasis weights (length = num_steps), higher = more structure early."""

    step_emphasis: Tuple[float, ...]


_ANTICIPATION = (
    r"\babout\s+to\b",
    r"\bon\s+the\s+verge\b",
    r"\bmoment\s+before\b",
    r"\bpoised\s+to\b",
)
_CLIMAX = (
    r"\bmid[- ]?air\b",
    r"\bat\s+the\s+peak\b",
    r"\bin\s+motion\b",
    r"\bsplashing\b",
)
_AFTERMATH = (
    r"\bjust\s+(?:after|before)\b",
    r"\baftermath\b",
    r"\bsettling\b",
    r"\bstill\s+smoking\b",
)


class TemporalMomentAnalyzer:
    """Classify temporal stance and emit denoise-step emphasis curve."""

    def __init__(self, num_steps: int = 28) -> None:
        self.num_steps = max(4, int(num_steps))

    def analyze(self, prompt: str) -> MomentCue:
        text = (prompt or "").lower()
        phase = MomentPhase.STATIC
        verb = ""
        confidence = 0.0

        for pat in _ANTICIPATION:
            m = re.search(pat, text)
            if m:
                phase = MomentPhase.ANTICIPATION
                verb = m.group(0)
                confidence = 0.85
                break
        if phase == MomentPhase.STATIC:
            for pat in _CLIMAX:
                m = re.search(pat, text)
                if m:
                    phase = MomentPhase.CLIMAX
                    verb = m.group(0)
                    confidence = 0.8
                    break
        if phase == MomentPhase.STATIC:
            for pat in _AFTERMATH:
                m = re.search(pat, text)
                if m:
                    phase = MomentPhase.AFTERMATH
                    verb = m.group(0)
                    confidence = 0.75
                    break

        frags: List[str] = []
        if phase == MomentPhase.ANTICIPATION:
            frags = ["tension in muscles", "frozen instant", "implied motion blur minimal"]
        elif phase == MomentPhase.CLIMAX:
            frags = ["dynamic motion", "action peak", "motion blur on extremities"]
        elif phase == MomentPhase.AFTERMATH:
            frags = ["debris settling", "trailing smoke", "relaxed poses"]

        return MomentCue(
            phase=phase,
            confidence=confidence,
            verb=verb,
            prompt_fragments=frags,
            step_emphasis=self._emphasis_curve(phase),
        )

    def _emphasis_curve(self, phase: MomentPhase) -> Tuple[float, ...]:
        n = self.num_steps
        if phase == MomentPhase.STATIC:
            return tuple(1.0 for _ in range(n))
        if phase == MomentPhase.ANTICIPATION:
            # structure early, leave noise late for "about to happen"
            return tuple(1.0 - 0.6 * (i / max(1, n - 1)) for i in range(n))
        if phase == MomentPhase.CLIMAX:
            mid = n // 2
            return tuple(0.7 + 0.6 * (1.0 - abs(i - mid) / max(1, mid)) for i in range(n))
        # aftermath — lock composition early, refine texture late
        return tuple(0.5 + 0.5 * (i / max(1, n - 1)) for i in range(n))
