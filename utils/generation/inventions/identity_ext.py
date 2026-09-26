"""Multi-character / identity inventions 60–70."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PerCharRefPlan",
    "plan_per_char_refs",
    "interaction_physics_addon",
    "identity_permanence_plan",
    "auto_character_sheet",
    "slot_dropout_caption",
    "cross_char_contrastive_loss",
    "cast_memory_update",
    "multi_char_conditioner_wiring_spec",
    "clothing_ownership_mask_spec",
    "voice_to_look_stub",
]


@dataclass
class PerCharRefPlan:
    refs: dict[str, str] = field(default_factory=dict)  # char_id -> image path
    argv_chunks: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_per_char_refs(refs: dict[str, str]) -> PerCharRefPlan:
    """Map character ids to InstantStyle reference paths (#60)."""
    chunks = []
    for cid, path in refs.items():
        if path:
            chunks.append(["--reference-image", path, "--reference-style-mode", "instantstyle"])
    return PerCharRefPlan(refs=dict(refs), argv_chunks=chunks, notes=[f"{len(refs)} char refs"])


def interaction_physics_addon(relation: str = "hugging") -> tuple[str, str]:
    table = {
        "hugging": ("plausible hug contact, arms wrap, no fused torsos", "merged bodies hugging error"),
        "fighting": ("clear fight spacing, impact contact, distinct silhouettes", "entangled limb spaghetti fight"),
        "holding_hands": ("hands clasped with correct finger interlock", "melted hand-holding"),
        "facing": ("facing each other, eye-line consistent", "gazes miss each other"),
    }
    return table.get(relation, table["facing"])


def identity_permanence_plan(character_id: str, sheet_path: str = "", face_ref: str = "") -> dict[str, Any]:
    return {
        "character_id": character_id,
        "lock": ["face", "hair", "body_silhouette"],
        "unlock": ["pose", "expression", "outfit_optional"],
        "sheet": sheet_path,
        "face_ref": face_ref,
        "argv": (["--character-sheet", sheet_path] if sheet_path else [])
        + (["--reference-image", face_ref] if face_ref else []),
    }


def auto_character_sheet(
    name: str,
    views: dict[str, str],
    *,
    out_path: str | Path,
) -> Path:
    """Build a character sheet JSON from front/side/back/detail view paths (#65)."""
    profile = {
        "character_name": name,
        "identity_tokens": [name],
        "reference_views": views,
        "face_features": [],
        "hair_features": [],
        "wardrobe": [],
        "pose_preferences": ["standing", "three-quarter view"],
        "avoid_tokens": ["face swap", "identity drift"],
        "prompt": f"consistent character {name}",
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return p


def slot_dropout_caption(blocks: list[str], *, drop_prob: float = 0.3, seed: int = 0) -> str:
    """Training augmentation (#68): randomly drop one character block."""
    import random

    r = random.Random(seed)
    if len(blocks) < 2 or r.random() > drop_prob:
        return ", ".join(blocks)
    keep = list(blocks)
    keep.pop(r.randrange(len(keep)))
    return ", ".join(keep)


def cross_char_contrastive_loss(sim_same: float, sim_other: float, margin: float = 0.2) -> float:
    """Moonshot training hook (#69): same-id similarity should exceed other-id."""
    return float(max(0.0, margin - float(sim_same) + float(sim_other)))


def cast_memory_update(memory: dict[str, Any], cast_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Session cast memory (#70)."""
    mem = dict(memory or {})
    casts = dict(mem.get("casts") or {})
    casts[cast_id] = payload
    mem["casts"] = casts
    return mem


def multi_char_conditioner_wiring_spec() -> dict[str, Any]:
    """Spec for wiring models.multi_character.MultiCharacterConditioner (#61)."""
    return {
        "module": "models.multi_character.MultiCharacterConditioner",
        "sample_hook": "inject CharacterSpec bboxes into cross-attn before CFG",
        "status": "scaffold",
        "cli": "--multi-char-conditioner",
    }


def clothing_ownership_mask_spec() -> dict[str, Any]:
    return {
        "idea": "mask color/clothing tokens to owner bbox in cross-attn",
        "status": "moonshot-scaffold",
        "depends_on": ["scene_graph", "cast_compiler"],
    }


def voice_to_look_stub(audio_embedding: list[float] | None = None) -> dict[str, Any]:
    return {
        "status": "stub",
        "mapping": "audio vibe → style tokens via learned projector",
        "dim": len(audio_embedding or []),
    }
