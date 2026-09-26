"""Multi-character cast: distinct outfits, poses, actions + anti-blend prompts.

Compiles a cast JSON into labeled positive/negative prompts, optional box layout
regions, and character-sheet style blocks — fighting identity/outfit bleed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.prompt.multi_subject import (
    MULTI_SUBJECT_NEGATIVE_EXTRAS,
    MULTI_SUBJECT_POSITIVE_EXTRAS,
    merge_character_sheet_negatives,
    merge_character_sheet_positives,
)

__all__ = [
    "CastMember",
    "CastScene",
    "load_cast_scene",
    "compile_cast_scene",
    "CastCompileResult",
]


@dataclass
class CastMember:
    id: str
    appearance: str = ""
    outfit: str = ""
    pose: str = ""
    action: str = ""
    spatial: str = ""  # left / right / center / bbox hint
    face_ref: str = ""
    sheet_path: str = ""
    avoid: list[str] = field(default_factory=list)

    def prompt_block(self) -> str:
        bits = [b for b in (self.appearance, self.outfit, self.pose, self.action, self.spatial) if b]
        return ", ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CastScene:
    members: list[CastMember] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    setting: str = ""
    quality: list[str] = field(default_factory=list)
    anti_blend: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "members": [m.to_dict() for m in self.members],
            "relations": list(self.relations),
            "setting": self.setting,
            "quality": list(self.quality),
            "anti_blend": self.anti_blend,
        }


@dataclass
class CastCompileResult:
    positive: str
    negative: str
    count_tag: str
    box_layout: dict[str, Any] | None
    sheet_paths: list[str]
    face_refs: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _member_from_obj(obj: dict[str, Any], fallback_id: str) -> CastMember:
    wardrobe = ", ".join(_as_list(obj.get("wardrobe") or obj.get("outfit") or obj.get("clothing")))
    pose = ", ".join(_as_list(obj.get("pose") or obj.get("poses")))
    action = ", ".join(_as_list(obj.get("action") or obj.get("actions") or obj.get("doing")))
    hair = ", ".join(_as_list(obj.get("hair") or obj.get("hair_features")))
    face = ", ".join(_as_list(obj.get("face") or obj.get("face_features") or obj.get("identity_tokens")))
    appearance = ", ".join(x for x in (_as_list(obj.get("appearance") or obj.get("description")) + [hair, face]) if x)
    spatial = str(obj.get("spatial_anchor") or obj.get("spatial") or obj.get("position") or "").strip()
    return CastMember(
        id=str(obj.get("id") or obj.get("name") or fallback_id),
        appearance=appearance,
        outfit=wardrobe,
        pose=pose,
        action=action,
        spatial=spatial,
        face_ref=str(obj.get("face_ref") or obj.get("reference_image") or "").strip(),
        sheet_path=str(obj.get("character_sheet") or obj.get("sheet") or "").strip(),
        avoid=_as_list(obj.get("avoid")),
    )


def load_cast_scene(path_or_dict: str | Path | dict[str, Any]) -> CastScene:
    if isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        data = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    actors = data.get("actors") or data.get("members") or data.get("characters") or []
    members = [_member_from_obj(a, f"character_{i + 1}") for i, a in enumerate(actors) if isinstance(a, dict)]
    rels: list[str] = []
    for r in data.get("relations") or []:
        if isinstance(r, str):
            rels.append(r)
        elif isinstance(r, dict):
            a, b = r.get("a"), r.get("b")
            kind = r.get("kind") or r.get("relation") or "interacting"
            detail = r.get("detail") or ""
            rels.append(f"{a} and {b}: {kind}" + (f", {detail}" if detail else ""))
    return CastScene(
        members=members,
        relations=rels,
        setting=str(data.get("setting") or data.get("place") or "").strip(),
        quality=_as_list(data.get("quality")),
        anti_blend=bool(data.get("anti_blend", data.get("anti_artifacts", True))),
    )


def _default_bboxes(n: int) -> list[tuple[float, float, float, float]]:
    """Non-overlapping horizontal strips for n characters."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.15, 0.1, 0.85, 0.95)]
    boxes = []
    w = 0.9 / n
    gap = 0.02
    for i in range(n):
        x0 = 0.05 + i * w + gap
        x1 = 0.05 + (i + 1) * w - gap
        boxes.append((x0, 0.08, min(x1, 0.95), 0.95))
    return boxes


def compile_cast_scene(scene: CastScene, *, base_prompt: str = "") -> CastCompileResult:
    n = len(scene.members)
    notes: list[str] = []
    if n == 0:
        return CastCompileResult(
            positive=base_prompt,
            negative="",
            count_tag="",
            box_layout=None,
            sheet_paths=[],
            face_refs=[],
            notes=["empty cast"],
        )

    # Count tags (booru-friendly)
    if n == 1:
        count = "1girl" if "girl" in (scene.members[0].appearance + scene.members[0].outfit).lower() else "1person"
    elif n == 2:
        count = "2girls"
    else:
        count = f"{n}girls" if n <= 6 else "multiple girls"
    # Prefer generic if mixed — keep simple count
    if n >= 2:
        count = f"{n}people" if n > 2 else "2people"
        # Also emit classic tags
        count = f"{n}girls" if n <= 4 else "multiple girls"

    labels = [m.id for m in scene.members]
    blocks = []
    for m in scene.members:
        block = m.prompt_block()
        # Explicit ownership phrasing fights clothing bleed
        owned = []
        if m.outfit:
            owned.append(f"{m.id} wears {m.outfit}")
        if m.action:
            owned.append(f"{m.id} is {m.action}")
        if m.pose:
            owned.append(f"{m.id} pose: {m.pose}")
        if owned:
            block = f"{block}, " + ", ".join(owned) if block else ", ".join(owned)
        blocks.append(block or m.id)

    labeled = merge_character_sheet_positives(blocks, labels=labels)
    parts = [count]
    if scene.quality:
        parts.append(", ".join(scene.quality))
    if base_prompt.strip():
        parts.append(base_prompt.strip().rstrip(","))
    parts.append(labeled)
    if scene.relations:
        parts.append(", ".join(scene.relations))
    if scene.setting:
        parts.append(scene.setting)
    if scene.anti_blend:
        parts.extend(MULTI_SUBJECT_POSITIVE_EXTRAS)
        notes.append("anti-blend positives on")

    positive = ", ".join(p for p in parts if p)

    neg_bits = list(MULTI_SUBJECT_NEGATIVE_EXTRAS) if scene.anti_blend else []
    neg_bits.extend(
        [
            "character blending",
            "face swap",
            "identity mix",
            "attributes swapped between characters",
            "shared clothing",
        ]
    )
    for m in scene.members:
        for a in m.avoid:
            neg_bits.append(a)
        # Explicit: don't put others' outfits on this character
        others_outfits = [o.outfit for o in scene.members if o.id != m.id and o.outfit]
        for oo in others_outfits[:2]:
            neg_bits.append(f"{m.id} wearing {oo}")
    negative = merge_character_sheet_negatives(neg_bits)

    # Box layout for regional CFG / layout attn
    boxes = _default_bboxes(n)
    regions = []
    for m, box in zip(scene.members, boxes, strict=False):
        regions.append(
            {
                "id": m.id,
                "prompt": m.prompt_block() or m.appearance or m.id,
                "box": [box[0], box[1], box[2], box[3]],
            }
        )
    box_layout = {
        "mode": "multi_character",
        "anti_bleed": True,
        "regions": regions,
        "global_prompt": scene.setting or base_prompt,
    }

    sheets = [m.sheet_path for m in scene.members if m.sheet_path]
    faces = [m.face_ref for m in scene.members if m.face_ref]
    notes.append(f"{n} characters compiled with ownership phrases")
    return CastCompileResult(
        positive=positive,
        negative=negative,
        count_tag=count,
        box_layout=box_layout,
        sheet_paths=sheets,
        face_refs=faces,
        notes=notes,
    )
