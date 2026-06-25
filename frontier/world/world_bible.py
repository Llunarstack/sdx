"""
World bible — continuity memory beyond a single prompt.

Product gap: Midjourney/Flux have character refs; few open stacks track *world state*
(name, outfit, location history) as first-class JSON the sampler consumes.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CharacterRecord:
    id: str
    display_name: str
    visual_tokens: str  # appended to prompt when character appears
    negative_tokens: str = ""
    reference_path: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class LocationRecord:
    id: str
    name: str
    visual_tokens: str
    palette: str = ""  # e.g. "teal shadows, amber highlights"
    tags: List[str] = field(default_factory=list)


@dataclass
class WorldLock:
    """Hard constraints for one generation."""

    character_ids: List[str] = field(default_factory=list)
    location_id: str = ""
    must_preserve: List[str] = field(default_factory=list)  # "red scarf", "clock tower"
    forbid: List[str] = field(default_factory=list)


@dataclass
class WorldBible:
    """Loadable world state for multi-shot / series generation."""

    world_id: str = "default"
    characters: Dict[str, CharacterRecord] = field(default_factory=dict)
    locations: Dict[str, LocationRecord] = field(default_factory=dict)
    lore: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "WorldBible":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        chars = {k: CharacterRecord(**v) for k, v in data.get("characters", {}).items()}
        locs = {k: LocationRecord(**v) for k, v in data.get("locations", {}).items()}
        return cls(
            world_id=data.get("world_id", p.stem),
            characters=chars,
            locations=locs,
            lore=data.get("lore", ""),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "world_id": self.world_id,
            "lore": self.lore,
            "characters": {k: asdict(v) for k, v in self.characters.items()},
            "locations": {k: asdict(v) for k, v in self.locations.items()},
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def detect_characters(self, prompt: str) -> List[CharacterRecord]:
        text = (prompt or "").lower()
        hits: List[CharacterRecord] = []
        for rec in self.characters.values():
            names = [rec.display_name.lower(), rec.id.lower(), *[t.lower() for t in rec.tags]]
            if any(n and n in text for n in names):
                hits.append(rec)
        return hits

    def detect_location(self, prompt: str) -> Optional[LocationRecord]:
        text = (prompt or "").lower()
        for rec in self.locations.values():
            if rec.name.lower() in text or rec.id.lower() in text:
                return rec
            if any(t.lower() in text for t in rec.tags):
                return rec
        return None

    def apply_lock(self, prompt: str, lock: WorldLock) -> str:
        parts = [prompt.strip()] if prompt.strip() else []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.visual_tokens:
                parts.append(rec.visual_tokens)
        loc = self.locations.get(lock.location_id)
        if loc and loc.visual_tokens:
            parts.append(loc.visual_tokens)
        if loc and loc.palette:
            parts.append(loc.palette)
        for item in lock.must_preserve:
            parts.append(f"keep {item} consistent")
        return ", ".join(parts)

    def negative_for_lock(self, lock: WorldLock) -> str:
        negs: List[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.negative_tokens:
                negs.append(rec.negative_tokens)
        for item in lock.forbid:
            negs.append(item)
        return ", ".join(negs)

    def references_for_lock(self, lock: WorldLock) -> List[str]:
        paths: List[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.reference_path:
                paths.append(rec.reference_path)
        return paths


_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")


def guess_proper_nouns(prompt: str) -> List[str]:
    """Cheap proper-noun harvest for auto world-bible seeding."""
    return list(dict.fromkeys(_NAME_RE.findall(prompt or "")))


__all__ = ["CharacterRecord", "LocationRecord", "WorldBible", "WorldLock", "guess_proper_nouns"]
