"""Palette harmony — complementary, analogous, triadic coherence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class HarmonyMode(str, Enum):
    COMPLEMENTARY = "complementary"
    ANALOGOUS = "analogous"
    TRIADIC = "triadic"
    MONOCHROME = "monochrome"
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class PaletteHarmony:
    mode: HarmonyMode
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, HarmonyMode, str, str], ...] = (
    (
        re.compile(r"\b(complementary|orange and teal|blue and orange)\b", re.I),
        HarmonyMode.COMPLEMENTARY,
        "complementary color tension, warm-cool separation in lights and shadows",
        "random hue noise, no color relationship",
    ),
    (
        re.compile(r"\b(analogous|harmonious palette|monochromatic blue)\b", re.I),
        HarmonyMode.ANALOGOUS,
        "analogous hue family, smooth temperature drift, restrained accent",
        "clashing unrelated hues",
    ),
    (
        re.compile(r"\b(triadic|three color|primary palette)\b", re.I),
        HarmonyMode.TRIADIC,
        "triadic balance with one dominant hue, two supporting accents",
        "equal loud primaries fighting for attention",
    ),
    (
        re.compile(r"\b(monochrome|black and white|sepia|single hue)\b", re.I),
        HarmonyMode.MONOCHROME,
        "monochrome value structure, subtle hue drift in shadows only",
        "color bleeding into monochrome",
    ),
    (
        re.compile(r"\b(warm palette|warm tones|golden warm)\b", re.I),
        HarmonyMode.WARM,
        "warm-dominant palette, cool only in shadows, inviting temperature",
        "cold blue cast on warm scene",
    ),
    (
        re.compile(r"\b(cool palette|cold tones|blue mood|teal grade)\b", re.I),
        HarmonyMode.COOL,
        "cool-dominant palette, warm accents only on skin or lights",
        "orange warm cast on cold scene",
    ),
)


class PalettePlanner:
    def plan(self, prompt: str) -> PaletteHarmony:
        text = prompt or ""
        for pat, mode, pos, neg in _RULES:
            if pat.search(text):
                return PaletteHarmony(mode, pos, neg)
        if re.search(r"\b(cinematic|color grade|film look)\b", text, re.I):
            return PaletteHarmony(
                HarmonyMode.NEUTRAL,
                "unified color grade, shadow hue separation, highlight rolloff",
                "uncorrected rainbow saturation, channel clipping",
            )
        return PaletteHarmony(HarmonyMode.NEUTRAL, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["HarmonyMode", "PaletteHarmony", "PalettePlanner"]
