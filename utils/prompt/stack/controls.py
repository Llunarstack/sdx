"""Resolve content-control kwargs from CLI + optional prompt inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt


def _get(args: Any, name: str, default: str = "none") -> str:
    if args is None:
        return default
    return str(getattr(args, name, default) or default)


def _get_bool(args: Any, name: str, default: bool = False) -> bool:
    if args is None:
        return default
    return bool(getattr(args, name, default))


@dataclass
class ContentControlState:
    """Explicit + inferred content-control modes (single source of truth)."""

    pose_mode: str = "none"
    view_angle: str = "none"
    subject_sex: str = "none"
    scene_domain: str = "none"
    clothing_mode: str = "none"
    background_mode: str = "none"
    people_layout: str = "none"
    relationship_mode: str = "none"
    object_layout: str = "none"
    hand_mode: str = "none"
    pose_naturalness: str = "none"
    typography_mode: str = "none"
    quality_pack: str = "none"
    lighting_mode: str = "none"
    skin_detail_mode: str = "none"
    body_proportion: str = "none"
    interaction_intensity: str = "none"
    advanced_pose: str = "none"
    object_interaction: str = "none"
    environment_type: str = "none"
    style_mode: str = "none"
    composition_mode: str = "none"
    artist_composition: str = "none"
    anti_ai_pack: str = "none"
    human_media_mode: str = "none"
    lora_scaffold: str = "none"
    adherence_pack: str = "none"
    # flags
    style_lock: bool = False
    anti_style_bleed: bool = False
    anti_duplicate_subjects: bool = False
    anti_perspective_drift: bool = False
    cleanup_conflicting_tags: bool = False
    allow_text_in_image: bool = False
    one_shot_boost: bool = True

    def to_apply_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def merge_content_control_overrides(state: ContentControlState, overrides: Mapping[str, Any]) -> ContentControlState:
    """Apply intelligence (or other) overrides without passing unknown keys to the dataclass."""
    if not overrides:
        return state
    data = state.to_apply_kwargs()
    valid = {f.name for f in fields(ContentControlState)}
    for key, val in overrides.items():
        if key not in valid or val is None:
            continue
        if isinstance(data.get(key), bool):
            if val:
                data[key] = bool(val)
        elif data.get(key) == "none" and val:
            data[key] = str(val)
    return ContentControlState(**data)


def resolve_content_controls(args: Any, prompt: str, *, auto_infer: bool = True) -> ContentControlState:
    """CLI flags (same names as the fields; ``--text-in-image`` -> ``allow_text_in_image``), then modes inferred
    from the prompt fill whatever the CLI left at ``"none"``."""
    data: dict[str, Any] = {}
    for f in fields(ContentControlState):
        src = "text_in_image" if f.name == "allow_text_in_image" else f.name
        data[f.name] = _get_bool(args, src, f.default) if isinstance(f.default, bool) else _get(args, src)
    if data["lora_scaffold"] == "none" and _get_bool(args, "lora_scaffold_auto") and getattr(args, "lora", None):
        data["lora_scaffold"] = "blend"
    state = ContentControlState(**data)
    if not auto_infer or args is None or not _get_bool(args, "auto_content_fix", True):
        return state
    return merge_content_control_overrides(state, infer_content_controls_from_prompt(prompt))


def apply_resolved_controls(positive: str, negative: str, state: ContentControlState) -> tuple[str, str]:
    return apply_content_controls(positive, negative, **state.to_apply_kwargs())
