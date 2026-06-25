"""Reusable motion packs — decouple motion from appearance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["MotionPack", "MotionLibrary", "parse_motion_library", "resolve_motion_clip"]


@dataclass(slots=True)
class MotionPack:
    id: str
    clip: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    camera: str = ""
    description: str = ""
    loop: bool = False


@dataclass(slots=True)
class MotionLibrary:
    packs: Dict[str, MotionPack] = field(default_factory=dict)

    def get(self, pack_id: str) -> Optional[MotionPack]:
        return self.packs.get(pack_id)


def parse_motion_library(raw: Any) -> MotionLibrary:
    lib = MotionLibrary()
    if not isinstance(raw, Mapping):
        return lib
    for k, v in raw.items():
        pid = str(k)
        if isinstance(v, str):
            lib.packs[pid] = MotionPack(id=pid, clip=v)
            continue
        if not isinstance(v, Mapping):
            continue
        tags = v.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        lib.packs[pid] = MotionPack(
            id=pid,
            clip=str(v.get("clip") or v.get("video") or v.get("path") or ""),
            tags=[str(t) for t in tags],
            category=str(v.get("category") or v.get("type") or "general"),
            camera=str(v.get("camera") or ""),
            description=str(v.get("description") or ""),
            loop=bool(v.get("loop")),
        )
    return lib


def resolve_motion_clip(
    library: MotionLibrary,
    pack_ids: Sequence[str],
    *,
    fallback: str = "",
) -> str:
    for pid in pack_ids:
        p = library.get(pid)
        if p and p.clip and Path(p.clip).is_file():
            return p.clip
    return fallback
