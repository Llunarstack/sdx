"""
Synesthesia engine — map sound/music/mood words to *generation knobs*, not just tags.

Outputs serendipity dial offset, CFG bias, color temperature hint — structurally different
from prompt-stack guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SynesthesiaTuning:
    mood: str
    serendipity_offset: float
    cfg_bias: float
    color_hint: str
    rhythm_hint: str


@dataclass(frozen=True)
class CrossModalMap:
    """Bundled tuning from audio/mood vocabulary."""

    tunings: Tuple[SynesthesiaTuning, ...]
    merged_color: str
    merged_rhythm: str


_AUDIO_MOOD: Tuple[Tuple[re.Pattern, str, float, float, str, str], ...] = (
    (
        re.compile(r"\b(jazz|swing|brass|smoky club)\b", re.I),
        "jazz",
        0.12,
        1.05,
        "amber highlights, blue shadow jazz",
        "syncopated visual rhythm, off-beat accents",
    ),
    (
        re.compile(r"\b(synthwave|retrowave|80s electronic|vaporwave)\b", re.I),
        "synthwave",
        0.08,
        1.0,
        "magenta-cyan gradient sky, neon grid",
        "pulse repetition, horizon line beat",
    ),
    (
        re.compile(r"\b(classical|orchestral|symphony|strings)\b", re.I),
        "classical",
        0.05,
        1.08,
        "balanced warm-cool orchestral grade",
        "crescendo toward focal point",
    ),
    (
        re.compile(r"\b(techno|industrial|distorted bass|rave)\b", re.I),
        "techno",
        0.18,
        0.95,
        "high contrast strobe accents",
        "staccato geometric repetition",
    ),
    (
        re.compile(r"\b(lullaby|gentle|soft piano|ambient)\b", re.I),
        "ambient",
        -0.05,
        1.02,
        "desaturated pastels, low contrast",
        "slow breathing negative space",
    ),
    (
        re.compile(r"\b(chaos|cacophony|dissonance|noise)\b", re.I),
        "chaos",
        0.25,
        0.92,
        "clashing hues with one anchor",
        "broken grid, visual dissonance",
    ),
)


class SynesthesiaEngine:
    def map_prompt(self, prompt: str) -> CrossModalMap:
        text = prompt or ""
        hits: list[SynesthesiaTuning] = []
        for pat, mood, ser_off, cfg, color, rhythm in _AUDIO_MOOD:
            if pat.search(text):
                hits.append(SynesthesiaTuning(mood, ser_off, cfg, color, rhythm))
        if not hits:
            return CrossModalMap((), "", "")
        colors = ", ".join(h.color_hint for h in hits[:2])
        rhythms = ", ".join(h.rhythm_hint for h in hits[:2])
        return CrossModalMap(tuple(hits), colors, rhythms)

    def diffusion_knobs(self, prompt: str) -> Tuple[float, float]:
        """Return (serendipity_offset, cfg_multiplier)."""
        m = self.map_prompt(prompt)
        if not m.tunings:
            return 0.0, 1.0
        ser = sum(t.serendipity_offset for t in m.tunings) / len(m.tunings)
        cfg = 1.0
        for t in m.tunings:
            cfg *= t.cfg_bias
        return ser, cfg


__all__ = ["CrossModalMap", "SynesthesiaEngine", "SynesthesiaTuning"]
