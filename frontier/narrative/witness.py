"""
Who is *looking* at the scene — not camera specs, but embodied viewpoint.

Maps narrative stance to prompt/camera fragments (child looking up, security feed, voyeur, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class WitnessPerspective(str, Enum):
    NEUTRAL = "neutral"
    CHILD = "child"  # low angle, wonder
    GUARD = "guard"  # surveillance, flat, wide
    LOVER = "lover"  # intimate close, soft bokeh
    STRANGER = "stranger"  # candid, off-center
    VICTIM = "victim"  # dutch angle, tension
    ARCHIVIST = "archivist"  # documentary, flat light


@dataclass(frozen=True)
class WitnessFrame:
    perspective: WitnessPerspective
    confidence: float
    prompt_fragments: List[str]
    cfg_bias: float  # multiply base CFG slightly


_WITNESS_TRIGGERS: Dict[WitnessPerspective, tuple[str, ...]] = {
    WitnessPerspective.CHILD: (
        r"\bchild'?s?\s+(?:view|eyes|perspective)\b",
        r"\bthrough\s+a\s+child\b",
        r"\bwonder\b",
    ),
    WitnessPerspective.GUARD: (
        r"\bsecurity\s+camera\b",
        r"\bCCTV\b",
        r"\bsurveillance\b",
        r"\bmonitor\s+feed\b",
    ),
    WitnessPerspective.LOVER: (
        r"\bintimate\b",
        r"\blover'?s?\s+gaze\b",
        r"\bfrom\s+the\s+bed\b",
    ),
    WitnessPerspective.STRANGER: (
        r"\bcandid\b",
        r"\bstreet\s+photography\b",
        r"\bunposed\b",
    ),
    WitnessPerspective.VICTIM: (
        r"\bthreatening\b",
        r"\bhorror\s+POV\b",
        r"\bfirst[- ]person\s+terror\b",
    ),
    WitnessPerspective.ARCHIVIST: (
        r"\bdocumentary\b",
        r"\barchival\b",
        r"\bhistorical\s+photo\b",
    ),
}

_FRAGMENT_HINTS: Dict[WitnessPerspective, tuple[str, ...]] = {
    WitnessPerspective.CHILD: ("low angle", "wide eyes in foreground optional", "soft natural light"),
    WitnessPerspective.GUARD: ("fisheye lens", "timestamp overlay", "high angle", "flat fluorescent"),
    WitnessPerspective.LOVER: ("shallow depth of field", "warm skin tones", "close framing"),
    WitnessPerspective.STRANGER: ("decisive moment", "off-center subject", "35mm lens"),
    WitnessPerspective.VICTIM: ("dutch angle", "motion blur edge", "claustrophobic framing"),
    WitnessPerspective.ARCHIVIST: ("neutral color grade", "even lighting", "medium format"),
}

_CFG_BIAS: Dict[WitnessPerspective, float] = {
    WitnessPerspective.NEUTRAL: 1.0,
    WitnessPerspective.CHILD: 1.05,
    WitnessPerspective.GUARD: 0.92,
    WitnessPerspective.LOVER: 1.08,
    WitnessPerspective.STRANGER: 1.0,
    WitnessPerspective.VICTIM: 1.12,
    WitnessPerspective.ARCHIVIST: 0.95,
}


class WitnessPerspectiveAnalyzer:
    """Infer embodied viewer from prompt cues."""

    def analyze(self, prompt: str) -> WitnessFrame:
        text = (prompt or "").strip()
        best = WitnessPerspective.NEUTRAL
        best_score = 0.0
        for perspective, patterns in _WITNESS_TRIGGERS.items():
            score = 0.0
            for pat in patterns:
                if re.search(pat, text, flags=re.IGNORECASE):
                    score += 1.0
            if score > best_score:
                best_score = score
                best = perspective

        confidence = min(1.0, best_score / 2.0) if best != WitnessPerspective.NEUTRAL else 0.0
        frags = list(_FRAGMENT_HINTS.get(best, ())) if confidence > 0 else []
        return WitnessFrame(
            perspective=best,
            confidence=confidence,
            prompt_fragments=frags,
            cfg_bias=_CFG_BIAS[best],
        )
