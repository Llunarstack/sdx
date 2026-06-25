"""
If the prompt implies an effect, flag missing causes (and vice versa).

Training-free guardrail: augment negative prompt or warn before wasting steps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class PlausibilityFlag:
    trigger: str
    missing: str
    severity: float  # 0..1
    category: str
    fix_hint: str


@dataclass(frozen=True)
class EffectImplication:
    """When ``cause`` appears, ``effect`` should appear (soft expectation)."""

    cause_pattern: str
    effect_pattern: str
    category: str
    fix_hint: str


# cause regex, expected effect regex, category, hint
_IMPLICATIONS: Tuple[EffectImplication, ...] = (
    EffectImplication(
        r"\brain\b|\bstorm\b|\bdownpour\b",
        r"\bwet\b|\bpuddle\b|\breflect(?:ion|ive)\b|\bglisten",
        "weather",
        "add wet surfaces or reflections",
    ),
    EffectImplication(
        r"\bsnow\b|\bblizzard\b",
        r"\bfrost\b|\bicy\b|\bcold\b|\bwhite\s+cover",
        "weather",
        "add frost, ice, or snow accumulation",
    ),
    EffectImplication(
        r"\bfire\b|\bflames\b|\bburning\b",
        r"\bsmoke\b|\bsoot\b|\bchar\b|\bglow\b|\bheat\s+haze",
        "physics",
        "add smoke, glow, or heat distortion",
    ),
    EffectImplication(
        r"\bunderwater\b|\bsubmerged\b",
        r"\bbubble\b|\bcaustic\b|\brefraction\b|\bfloating\s+hair",
        "medium",
        "add underwater cues (caustics, bubbles, hair float)",
    ),
    EffectImplication(
        r"\bnight\b|\bmidnight\b",
        r"\bdark\b|\bshadow\b|\bartificial\s+light\b|\bstreetlamp\b|\bmoon",
        "lighting",
        "add night lighting sources or deep shadows",
    ),
    EffectImplication(
        r"\brunning\b|\bsprint\b|\bmotion\s+blur\b",
        r"\bblur\b|\bdynamic\b|\bwind\b|\bstrain\b",
        "motion",
        "add motion blur or dynamic pose cues",
    ),
    EffectImplication(
        r"\bbroken\b|\bshattered\b|\bcrack",
        r"\bdebris\b|\bfragment\b|\bjagged\b",
        "damage",
        "add debris or jagged edges",
    ),
)


class PhysicalPlausibilityScanner:
    """Scan prompts for missing implied effects."""

    def __init__(self, rules: Sequence[EffectImplication] | None = None) -> None:
        self.rules = tuple(rules or _IMPLICATIONS)

    def scan(self, prompt: str) -> List[PlausibilityFlag]:
        text = (prompt or "").strip()
        if not text:
            return []
        flags: List[PlausibilityFlag] = []
        for rule in self.rules:
            cause = _first_match(text, rule.cause_pattern)
            if cause is None:
                continue
            effect = _first_match(text, rule.effect_pattern)
            if effect is not None:
                continue
            flags.append(
                PlausibilityFlag(
                    trigger=cause,
                    missing=rule.effect_pattern,
                    severity=0.55,
                    category=rule.category,
                    fix_hint=rule.fix_hint,
                )
            )
        return flags

    def augment_prompt(self, prompt: str, flags: Sequence[PlausibilityFlag], max_add: int = 2) -> str:
        """Append cheap fix hints as comma-separated fragments."""
        if not flags:
            return prompt
        hints = [f.fix_hint for f in flags[:max_add]]
        return f"{prompt}, {', '.join(hints)}" if prompt else ", ".join(hints)

    def negative_suffix(self, flags: Sequence[PlausibilityFlag]) -> str:
        """Generic negatives when physics cues are absent."""
        if not flags:
            return ""
        parts = ["physically impossible", "floating without support", "wrong lighting for weather"]
        return ", ".join(parts[: min(3, len(flags))])


def _first_match(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


__all__ = ["EffectImplication", "PhysicalPlausibilityScanner", "PlausibilityFlag"]
