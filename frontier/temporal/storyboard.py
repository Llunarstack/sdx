"""
Storyboard chain — sequence of prompts with carry-over locks.

Video models get temporal consistency; still-image stacks rarely expose *series* mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from frontier.world.world_bible import WorldLock


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
    beats: List[StoryboardBeat] = field(default_factory=list)
    carry_tags: List[str] = field(default_factory=list)  # appended to every beat after first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "carry_tags": self.carry_tags,
            "beats": [asdict(b) for b in self.beats],
        }

    @classmethod
    def load(cls, path: str | Path) -> "Storyboard":
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

    def expanded_prompts(self) -> List[str]:
        carry = ", ".join(self.carry_tags) if self.carry_tags else ""
        out: List[str] = []
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
