"""Session memory box for still-image agentic generation.

Stores facets, RAG notes, per-facet refs, InstantStyle/LoRA choices, and
clarify answers across iterative generate/edit loops.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["MemoryBox", "load_memory_box", "save_memory_box"]


@dataclass
class MemoryBox:
    session_id: str
    prompt: str = ""
    facets: dict[str, Any] = field(default_factory=dict)
    clarify: dict[str, Any] = field(default_factory=dict)
    rag_notes: list[str] = field(default_factory=list)
    refs: dict[str, list[str]] = field(default_factory=dict)
    style_mode: str = "instantstyle"
    lora_path: str = ""
    character_sheet: str = ""
    last_metrics: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())

    def remember(self, event: str, **payload: Any) -> None:
        self.history.append({"t": time.time(), "event": event, **payload})
        self.updated_at = time.time()
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def set_refs(self, facet: str, paths: list[str]) -> None:
        self.refs[str(facet)] = [str(p) for p in paths]
        self.remember("refs", facet=facet, n=len(paths))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryBox:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})


def load_memory_box(path: str | Path) -> MemoryBox | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return MemoryBox.from_dict(data)
    except Exception:
        return None


def save_memory_box(box: MemoryBox, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    box.updated_at = time.time()
    p.write_text(json.dumps(box.to_dict(), indent=2), encoding="utf-8")
    return p
