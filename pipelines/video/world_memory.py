"""World bible — persistent setting rules across generations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

__all__ = ["WorldBible", "parse_world", "world_to_prompt", "world_negative"]


@dataclass(slots=True)
class WorldBible:
    id: str
    name: str = ""
    architecture: str = ""
    technology: str = ""
    weather: str = ""
    palette: str = ""
    culture: str = ""
    geography: str = ""
    magic_rules: str = ""
    era: str = ""
    locations: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


def parse_world(raw: Any) -> WorldBible | None:
    if not raw:
        return None
    if isinstance(raw, str):
        return WorldBible(id="world", name=raw)
    if not isinstance(raw, Mapping):
        return None
    locs = raw.get("locations") or {}
    if not isinstance(locs, Mapping):
        locs = {}
    wid = str(raw.get("id") or "world")
    magic = raw.get("magic")
    magic_s = ""
    if isinstance(magic, Mapping):
        magic_s = ", ".join(f"{k}: {v}" for k, v in magic.items())
    elif isinstance(magic, str):
        magic_s = magic
    return WorldBible(
        id=wid,
        name=str(raw.get("name") or raw.get("title") or wid),
        architecture=str(raw.get("architecture") or ""),
        technology=str(raw.get("technology") or raw.get("tech") or ""),
        weather=str(raw.get("weather") or raw.get("climate") or ""),
        palette=str(raw.get("palette") or raw.get("colors") or ""),
        culture=str(raw.get("culture") or ""),
        geography=str(raw.get("geography") or ""),
        magic_rules=magic_s,
        era=str(raw.get("era") or ""),
        locations={str(k): str(v) for k, v in locs.items()},
        notes=str(raw.get("notes") or raw.get("description") or ""),
    )


def world_to_prompt(w: WorldBible, *, location: str = "") -> str:
    parts: List[str] = [f"world {w.name}" if w.name else "consistent world"]
    for label, val in (
        ("architecture", w.architecture),
        ("technology", w.technology),
        ("weather", w.weather),
        ("palette", w.palette),
        ("culture", w.culture),
        ("geography", w.geography),
        ("era", w.era),
        ("magic", w.magic_rules),
    ):
        if val:
            parts.append(f"{label}: {val}")
    if location and location in w.locations:
        parts.append(f"location {location}: {w.locations[location]}")
    if w.notes:
        parts.append(w.notes)
    return ", ".join(parts)


def world_negative(w: WorldBible) -> str:
    parts = ["anachronistic technology mismatch", "wrong architectural style"]
    if w.palette:
        parts.append("off-palette colors")
    return ", ".join(parts)
