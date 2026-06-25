"""Atmospheric depth — fog, haze, god rays, weather mood."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class WeatherMood(str, Enum):
    FOG = "fog"
    MIST = "mist"
    RAIN = "rain"
    SNOW = "snow"
    STORM = "storm"
    CLEAR = "clear"
    GOD_RAYS = "god_rays"
    DUST = "dust"


@dataclass(frozen=True)
class AtmospherePlan:
    mood: WeatherMood
    positive: str
    negative: str


_RULES: Tuple[Tuple[re.Pattern, WeatherMood, str, str], ...] = (
    (
        re.compile(r"\b(fog|foggy|misty city|pea soup)\b", re.I),
        WeatherMood.FOG,
        "volumetric fog depth falloff, soft aerial perspective, light scatter",
        "uniform gray overlay, no depth falloff",
    ),
    (
        re.compile(r"\b(god rays|crepuscular|light shafts|volumetric light)\b", re.I),
        WeatherMood.GOD_RAYS,
        "crepuscular god rays through particulate, motivated beam direction",
        "random lens flare streaks without volume",
    ),
    (
        re.compile(r"\b(rain|rainy|downpour|wet street)\b", re.I),
        WeatherMood.RAIN,
        "rain streaks, wet surface reflections, micro ripples, moody atmosphere",
        "dry pavement in rain, plastic rain overlay",
    ),
    (
        re.compile(r"\b(snow|blizzard|snowfall|winter storm)\b", re.I),
        WeatherMood.SNOW,
        "snow particle depth, accumulation on surfaces, cold breath optional",
        "white noise overlay, no surface accumulation",
    ),
    (
        re.compile(r"\b(dust|sandstorm|haze desert|smoke filled)\b", re.I),
        WeatherMood.DUST,
        "particulate haze, warm dust scatter, reduced distant contrast",
        "clean air in dust storm",
    ),
)


class AtmospherePlanner:
    def plan(self, prompt: str) -> AtmospherePlan:
        text = prompt or ""
        for pat, mood, pos, neg in _RULES:
            if pat.search(text):
                return AtmospherePlan(mood, pos, neg)
        if re.search(r"\b(landscape|exterior|outdoor|city)\b", text, re.I):
            return AtmospherePlan(
                WeatherMood.CLEAR,
                "atmospheric perspective, distant desaturation, aerial depth",
                "flat depth, everything same sharpness",
            )
        return AtmospherePlan(WeatherMood.CLEAR, "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.plan(prompt)
        return p.positive, p.negative


__all__ = ["AtmospherePlan", "AtmospherePlanner", "WeatherMood"]
