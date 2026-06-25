"""Director personality presets — Spielberg, Nolan, Miyazaki, etc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["DirectorPersonality", "director_by_id", "list_directors"]


@dataclass(frozen=True, slots=True)
class DirectorPersonality:
    id: str
    title: str
    positive: str
    negative: str
    pacing: str
    default_camera: str


_DIRECTORS: Dict[str, DirectorPersonality] = {
    "spielberg": DirectorPersonality(
        "spielberg",
        "Spielberg",
        "wonder, backlight silhouettes, emotional blocking, wide wonder shots",
        "cynical flat staging",
        "measured wonder builds",
        "slow dolly inward",
    ),
    "nolan": DirectorPersonality(
        "nolan",
        "Nolan",
        "IMAX scale, practical weight, non-linear tension, architectural symmetry",
        "handheld chaos, cartoon physics",
        "deliberate escalation",
        "crane wide",
    ),
    "tarantino": DirectorPersonality(
        "tarantino",
        "Tarantino",
        "dialogue tension, trunk shots, bold color, long takes",
        "generic coverage",
        "dialogue-heavy holds",
        "low angle",
    ),
    "miyazaki": DirectorPersonality(
        "miyazaki",
        "Miyazaki",
        "quiet environmental beauty, gentle pacing, wind in grass, lived-in worlds",
        "aggressive shaky cam, harsh cuts",
        "contemplative",
        "pan across landscape",
    ),
    "anime_action": DirectorPersonality(
        "anime_action",
        "Anime action director",
        "speed lines, impact frames, dramatic angles, held reaction shots",
        "muddy coverage",
        "beat on impact",
        "snap zoom",
    ),
    "horror": DirectorPersonality(
        "horror",
        "Horror director",
        "negative space, slow dread, off-screen threat, underexposed corners",
        "bright flat comedy lighting",
        "slow reveal",
        "static wide",
    ),
}


def list_directors() -> list[DirectorPersonality]:
    return list(_DIRECTORS.values())


def director_by_id(name: str) -> Optional[DirectorPersonality]:
    return _DIRECTORS.get((name or "").strip().lower().replace(" ", "_"))
