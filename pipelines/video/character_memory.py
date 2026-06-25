"""Persistent character bible — identity across episodes and styles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

__all__ = ["CharacterBible", "parse_character_bibles", "bible_to_prompt", "bible_negative"]


@dataclass(slots=True)
class CharacterBible:
    id: str
    name: str = ""
    hair: str = ""
    eyes: str = ""
    height: str = ""
    build: str = ""
    outfit: str = ""
    voice: str = ""
    walk_style: str = ""
    scars: str = ""
    age: str = ""
    emotion_baseline: str = ""
    reference_images: List[str] = field(default_factory=list)
    notes: str = ""


def parse_character_bibles(raw: Any) -> Dict[str, CharacterBible]:
    out: Dict[str, CharacterBible] = {}
    if not isinstance(raw, Mapping):
        return out
    for k, v in raw.items():
        cid = str(k)
        if isinstance(v, str):
            out[cid] = CharacterBible(id=cid, name=v)
            continue
        if not isinstance(v, Mapping):
            continue
        imgs = v.get("images") or v.get("refs") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        out[cid] = CharacterBible(
            id=cid,
            name=str(v.get("name") or cid),
            hair=str(v.get("hair") or ""),
            eyes=str(v.get("eyes") or ""),
            height=str(v.get("height") or ""),
            build=str(v.get("build") or ""),
            outfit=str(v.get("outfit") or v.get("armor") or v.get("costume") or ""),
            voice=str(v.get("voice") or ""),
            walk_style=str(v.get("walk_style") or v.get("gait") or ""),
            scars=str(v.get("scars") or ""),
            age=str(v.get("age") or ""),
            emotion_baseline=str(v.get("emotion") or v.get("mood") or ""),
            reference_images=[str(x) for x in imgs],
            notes=str(v.get("notes") or v.get("description") or ""),
        )
    return out


def bible_to_prompt(b: CharacterBible) -> str:
    parts: List[str] = []
    if b.name:
        parts.append(f"character {b.name}")
    for label, val in (
        ("hair", b.hair),
        ("eyes", b.eyes),
        ("height", b.height),
        ("build", b.build),
        ("outfit", b.outfit),
        ("walk style", b.walk_style),
        ("age", b.age),
        ("scars", b.scars),
        ("baseline emotion", b.emotion_baseline),
    ):
        if val:
            parts.append(f"{label}: {val}")
    if b.notes:
        parts.append(b.notes)
    return ", ".join(parts)


def bible_negative(b: CharacterBible) -> str:
    base = "wrong face, duplicate head, morphing identity"
    if b.hair:
        base += f", wrong hair color, not {b.hair}"
    if b.eyes:
        base += ", wrong eye color"
    return base
