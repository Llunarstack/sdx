"""
Mood physics — emotional adjectives tune serendipity, CFG, and step emphasis.

Unlike static negative prompts: outputs numeric knobs for the sampler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class MoodVector:
    name: str
    serendipity: float  # -0.1 .. +0.3 offset
    cfg_mult: float
    step_emphasis_early: float  # weight early vs late structure
    fragment: str


@dataclass(frozen=True)
class MoodPhysicsPlan:
    vectors: Tuple[MoodVector, ...]
    serendipity_offset: float
    cfg_mult: float
    prompt_fragment: str


_VECTORS: Tuple[Tuple[re.Pattern, str, float, float, float, str], ...] = (
    (
        re.compile(r"\b(ominous|dread|foreboding|eerie)\b", re.I),
        "ominous",
        0.05,
        1.1,
        0.7,
        "low-key tension, withheld reveal",
    ),
    (
        re.compile(r"\b(joyful|euphoric|celebration|festive)\b", re.I),
        "joyful",
        0.15,
        1.0,
        0.4,
        "bright energy, open gesture",
    ),
    (
        re.compile(r"\b(melancholic|lonely|wistful|sad)\b", re.I),
        "melancholic",
        -0.02,
        1.05,
        0.55,
        "quiet emptiness, soft desaturation",
    ),
    (
        re.compile(r"\b(manic|frenetic|chaotic energy)\b", re.I),
        "manic",
        0.22,
        0.94,
        0.35,
        "visual stutter, high entropy",
    ),
    (re.compile(r"\b(serene|calm|peaceful|zen)\b", re.I), "serene", -0.08, 1.03, 0.6, "stillness, balanced masses"),
    (re.compile(r"\b(romantic|tender|intimate)\b", re.I), "romantic", 0.03, 1.06, 0.5, "soft proximity, warm falloff"),
    (
        re.compile(r"\b(awe|sublime|transcendent|majestic)\b", re.I),
        "awe",
        0.06,
        1.12,
        0.75,
        "scale overwhelm, vastness",
    ),
)


class MoodPhysics:
    def analyze(self, prompt: str) -> MoodPhysicsPlan:
        text = prompt or ""
        hits: List[MoodVector] = []
        for pat, name, ser, cfg, early, frag in _VECTORS:
            if pat.search(text):
                hits.append(MoodVector(name, ser, cfg, early, frag))
        if not hits:
            return MoodPhysicsPlan((), 0.0, 1.0, "")
        ser = sum(h.serendipity for h in hits) / len(hits)
        cfg = 1.0
        for h in hits:
            cfg *= h.cfg_mult
        frags = ", ".join(h.fragment for h in hits[:2])
        return MoodPhysicsPlan(tuple(hits), ser, cfg, frags)

    def step_emphasis_curve(self, plan: MoodPhysicsPlan, num_steps: int) -> Tuple[float, ...]:
        """Early-heavy vs late-heavy emphasis from mood."""
        if not plan.vectors:
            return tuple(1.0 for _ in range(num_steps))
        early_w = sum(v.step_emphasis_early for v in plan.vectors) / len(plan.vectors)
        n = max(4, num_steps)
        curve: List[float] = []
        for i in range(n):
            t = i / max(1, n - 1)
            w = (1.0 - t) * early_w + t * (1.0 - early_w)
            curve.append(0.85 + 0.3 * w)
        return tuple(curve)


__all__ = ["MoodPhysics", "MoodPhysicsPlan", "MoodVector"]
