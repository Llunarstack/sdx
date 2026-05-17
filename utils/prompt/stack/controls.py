"""Resolve content-control kwargs from CLI + optional prompt inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping

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

    def to_apply_kwargs(self) -> Dict[str, Any]:
        return asdict(self)


_INFER_KEYS = frozenset(
    f.name
    for f in fields(ContentControlState)
    if f.name
    not in {
        "style_lock",
        "anti_style_bleed",
        "anti_duplicate_subjects",
        "anti_perspective_drift",
        "cleanup_conflicting_tags",
        "allow_text_in_image",
        "one_shot_boost",
        "subject_sex",
        "background_mode",
        "relationship_mode",
        "object_layout",
        "pose_naturalness",
        "typography_mode",
        "skin_detail_mode",
        "body_proportion",
        "interaction_intensity",
        "advanced_pose",
        "object_interaction",
        "environment_type",
        "lora_scaffold",
    }
)


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
    """Merge argparse flags with optional ``infer_content_controls_from_prompt``."""
    lora_scaffold = _get(args, "lora_scaffold")
    if (
        args is not None
        and _get_bool(args, "lora_scaffold_auto")
        and getattr(args, "lora", None)
        and lora_scaffold == "none"
    ):
        lora_scaffold = "blend"

    state = ContentControlState(
        pose_mode=_get(args, "pose_mode"),
        view_angle=_get(args, "view_angle"),
        subject_sex=_get(args, "subject_sex"),
        scene_domain=_get(args, "scene_domain"),
        clothing_mode=_get(args, "clothing_mode"),
        background_mode=_get(args, "background_mode"),
        people_layout=_get(args, "people_layout"),
        relationship_mode=_get(args, "relationship_mode"),
        object_layout=_get(args, "object_layout"),
        hand_mode=_get(args, "hand_mode"),
        pose_naturalness=_get(args, "pose_naturalness"),
        typography_mode=_get(args, "typography_mode"),
        quality_pack=_get(args, "quality_pack"),
        lighting_mode=_get(args, "lighting_mode"),
        skin_detail_mode=_get(args, "skin_detail_mode"),
        body_proportion=_get(args, "body_proportion"),
        interaction_intensity=_get(args, "interaction_intensity"),
        advanced_pose=_get(args, "advanced_pose", "none") if args is not None else "none",
        object_interaction=_get(args, "object_interaction", "none") if args is not None else "none",
        environment_type=_get(args, "environment_type", "none") if args is not None else "none",
        style_mode=_get(args, "style_mode"),
        composition_mode=_get(args, "composition_mode"),
        artist_composition=_get(args, "artist_composition"),
        anti_ai_pack=_get(args, "anti_ai_pack"),
        human_media_mode=_get(args, "human_media_mode"),
        lora_scaffold=lora_scaffold,
        adherence_pack=_get(args, "adherence_pack"),
        style_lock=_get_bool(args, "style_lock"),
        anti_style_bleed=_get_bool(args, "anti_style_bleed"),
        anti_duplicate_subjects=_get_bool(args, "anti_duplicate_subjects"),
        anti_perspective_drift=_get_bool(args, "anti_perspective_drift"),
        cleanup_conflicting_tags=_get_bool(args, "cleanup_conflicting_tags"),
        allow_text_in_image=_get_bool(args, "text_in_image"),
        one_shot_boost=_get_bool(args, "one_shot_boost", True),
    )

    if not auto_infer or args is None or not _get_bool(args, "auto_content_fix", True):
        return state

    inferred = infer_content_controls_from_prompt(prompt)
    data = asdict(state)
    for key in _INFER_KEYS:
        if data.get(key) == "none" and inferred.get(key):
            data[key] = str(inferred[key])
    return ContentControlState(**data)


def apply_resolved_controls(positive: str, negative: str, state: ContentControlState) -> tuple[str, str]:
    return apply_content_controls(positive, negative, **state.to_apply_kwargs())
