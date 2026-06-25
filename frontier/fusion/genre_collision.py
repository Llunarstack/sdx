"""
Genre fusion — steampunk+cyberpunk, folk+sci-fi without muddy soup.

Picks a *dominant* genre and treats the other as accent (30% rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class GenrePair(str, Enum):
    STEAMPUNK_CYBER = "steampunk_cyber"
    FOLK_SCIFI = "folk_scifi"
    ANIME_OIL = "anime_oil"
    BRUTALIST_ORGANIC = "brutalist_organic"
    WESTERN_FANTASY = "western_fantasy"
    NONE = "none"


@dataclass(frozen=True)
class FusionPlan:
    pair: GenrePair
    dominant: str
    accent: str
    positive: str
    negative: str


_DETECT: Tuple[Tuple[str, str, GenrePair, str, str, str, str], ...] = (
    (
        r"steampunk",
        r"cyberpunk",
        GenrePair.STEAMPUNK_CYBER,
        "steampunk",
        "cyberpunk",
        "brass Victorian engineering dominant, neon cyber accents at 30%, unified grime palette",
        "equal mix mush, chrome on brass without logic",
    ),
    (
        r"folk",
        r"sci[- ]?fi|space|astronaut",
        GenrePair.FOLK_SCIFI,
        "folk art",
        "sci-fi",
        "folk pattern and handcraft dominant, sci-fi element as single focal artifact",
        "random spacesuit in medieval tapestry",
    ),
    (
        r"anime",
        r"oil painting|impasto",
        GenrePair.ANIME_OIL,
        "anime",
        "oil painting",
        "anime linework with oil impasto fill, cel edges over painterly body",
        "flat anime with fake oil filter",
    ),
    (
        r"brutalist|concrete",
        r"organic|biomorphic|vine",
        GenrePair.BRUTALIST_ORGANIC,
        "brutalist",
        "organic",
        "concrete mass dominant, biomorphic breach as singular rupture",
        "ivy wallpaper on brutalism",
    ),
    (
        r"western|cowboy",
        r"fantasy|dragon|magic",
        GenrePair.WESTERN_FANTASY,
        "western",
        "fantasy",
        "frontier western dominant, one fantasy element as mythic intrusion",
        "elf cowboy costume party",
    ),
)


class GenreCollisionEngine:
    def detect(self, prompt: str) -> FusionPlan:
        text = (prompt or "").lower()
        for a_pat, b_pat, pair, dom, acc, pos, neg in _DETECT:
            if re.search(a_pat, text, re.I) and re.search(b_pat, text, re.I):
                return FusionPlan(pair, dom, acc, pos, neg)
        return FusionPlan(GenrePair.NONE, "", "", "", "")

    def fragments(self, prompt: str) -> Tuple[str, str]:
        p = self.detect(prompt)
        return p.positive, p.negative


__all__ = ["FusionPlan", "GenreCollisionEngine", "GenrePair"]
