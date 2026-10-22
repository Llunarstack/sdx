"""
Storyboard chain — sequence of prompts with carry-over locks.

Video models get temporal consistency; still-image stacks rarely expose *series* mode.
"""

from __future__ import annotations

import json

# World bible: characters / locations / locks the storyboard keeps consistent across beats.
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CharacterRecord:
    id: str
    display_name: str
    visual_tokens: str  # appended to prompt when character appears
    negative_tokens: str = ""
    reference_path: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class LocationRecord:
    id: str
    name: str
    visual_tokens: str
    palette: str = ""  # e.g. "teal shadows, amber highlights"
    tags: list[str] = field(default_factory=list)


@dataclass
class WorldBible:
    """Loadable world state for multi-shot / series generation."""

    world_id: str = "default"
    characters: dict[str, CharacterRecord] = field(default_factory=dict)
    locations: dict[str, LocationRecord] = field(default_factory=dict)
    lore: str = ""

    @classmethod
    def load(cls, path: str | Path) -> WorldBible:
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

    def detect_characters(self, prompt: str) -> list[CharacterRecord]:
        text = (prompt or "").lower()
        hits: list[CharacterRecord] = []
        for rec in self.characters.values():
            names = [rec.display_name.lower(), rec.id.lower(), *[t.lower() for t in rec.tags]]
            if any(n and n in text for n in names):
                hits.append(rec)
        return hits

    def detect_location(self, prompt: str) -> LocationRecord | None:
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
        negs: list[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.negative_tokens:
                negs.append(rec.negative_tokens)
        for item in lock.forbid:
            negs.append(item)
        return ", ".join(negs)

    def references_for_lock(self, lock: WorldLock) -> list[str]:
        paths: list[str] = []
        for cid in lock.character_ids:
            rec = self.characters.get(cid)
            if rec and rec.reference_path:
                paths.append(rec.reference_path)
        return paths


_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")


def guess_proper_nouns(prompt: str) -> list[str]:
    """Cheap proper-noun harvest for auto world-bible seeding."""
    return list(dict.fromkeys(_NAME_RE.findall(prompt or "")))


@dataclass
class WorldLock:
    """Hard constraints for one generation."""

    character_ids: list[str] = field(default_factory=list)
    location_id: str = ""
    must_preserve: list[str] = field(default_factory=list)  # "red scarf", "clock tower"
    forbid: list[str] = field(default_factory=list)


@dataclass
class StoryboardBeat:
    index: int
    prompt: str
    negative_prompt: str = ""
    lock: WorldLock = field(default_factory=WorldLock)
    seed_offset: int = 0
    notes: str = ""


@dataclass
class Storyboard:
    title: str = ""
    beats: list[StoryboardBeat] = field(default_factory=list)
    carry_tags: list[str] = field(default_factory=list)  # appended to every beat after first

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "carry_tags": self.carry_tags,
            "beats": [asdict(b) for b in self.beats],
        }

    @classmethod
    def load(cls, path: str | Path) -> Storyboard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        beats = []
        for i, b in enumerate(data.get("beats", [])):
            lock = WorldLock(**b.get("lock", {}))
            beats.append(
                StoryboardBeat(
                    index=int(b.get("index", i)),
                    prompt=b.get("prompt", ""),
                    negative_prompt=b.get("negative_prompt", ""),
                    lock=lock,
                    seed_offset=int(b.get("seed_offset", 0)),
                    notes=b.get("notes", ""),
                )
            )
        return cls(title=data.get("title", ""), beats=beats, carry_tags=list(data.get("carry_tags", [])))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def expanded_prompts(self) -> list[str]:
        carry = ", ".join(self.carry_tags) if self.carry_tags else ""
        out: list[str] = []
        for i, beat in enumerate(self.beats):
            p = beat.prompt
            if i > 0 and carry:
                p = f"{p}, {carry}" if p else carry
            out.append(p)
        return out


class StoryboardPlanner:
    """Build storyboards from bullet lists or numbered scenes."""

    def from_scene_list(self, scenes: Sequence[str], *, title: str = "") -> Storyboard:
        beats = [StoryboardBeat(index=i, prompt=s.strip(), seed_offset=i) for i, s in enumerate(scenes) if s.strip()]
        return Storyboard(title=title, beats=beats)

    def with_carry(self, board: Storyboard, tags: Sequence[str]) -> Storyboard:
        board.carry_tags = list(tags)
        return board


__all__ = ["Storyboard", "StoryboardBeat", "StoryboardPlanner"]
