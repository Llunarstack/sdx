"""
Tag lists/dicts for ``content_controls``, loaded from ``data/prompt_tags/*.csv``.

Regenerate CSVs from the repo root::

    python -m scripts.tools dump_prompt_tag_csvs
"""

from __future__ import annotations

# ``from .content_control_tags import *`` only re-exports private names if listed here.
__all__ = (
    "_ADHERENCE_PACK_NEGATIVE",
    "_ADHERENCE_PACK_POSITIVE",
    "_ADHERENCE_STANDARD_NEGATIVE",
    "_ADHERENCE_STANDARD_POSITIVE",
    "_ADHERENCE_STRICT_EXTRA_NEGATIVE",
    "_ADHERENCE_STRICT_EXTRA_POSITIVE",
    "_ADVANCED_POSE_POSITIVE",
    "_ANGLE_NEGATIVE",
    "_ANGLE_POSITIVE",
    "_ARTIST_COMPOSITION_NEGATIVE",
    "_ARTIST_COMPOSITION_POSITIVE",
    "_ANTI_AI_LITE_NEGATIVE",
    "_ANTI_AI_LITE_POSITIVE",
    "_ANTI_AI_STRONG_EXTRA_NEGATIVE",
    "_ANTI_AI_STRONG_EXTRA_POSITIVE",
    "_ANTI_AI_STRONG_NEGATIVE",
    "_ANTI_AI_STRONG_POSITIVE",
    "_BACKGROUND_NEGATIVE",
    "_BACKGROUND_POSITIVE",
    "_BEST_MODEL_TAGS",
    "_BODY_PROPORTION_POSITIVE",
    "_CLOTHING_NEGATIVE",
    "_CLOTHING_POSITIVE",
    "_COMPOSITION_NEGATIVE",
    "_COMPOSITION_POSITIVE",
    "_CONFLICTING_TAG_PAIRS",
    "_DOMAIN_NEGATIVE",
    "_DOMAIN_POSITIVE",
    "_DUPLICATE_SUBJECT_NEGATIVE",
    "_ENVIRONMENT_DETAIL_POSITIVE",
    "_HAND_NEGATIVE",
    "_HAND_POSITIVE",
    "_HUMAN_MEDIA_NEGATIVE",
    "_HUMAN_MEDIA_POSITIVE",
    "_INTERACTION_INTENSITY",
    "_LIGHTING_MODE_NEGATIVE",
    "_LIGHTING_MODE_POSITIVE",
    "_LORA_SCAFFOLD_NEGATIVE",
    "_LORA_SCAFFOLD_POSITIVE",
    "_NSFW_NEGATIVE",
    "_NSFW_PACK_NEGATIVE",
    "_NSFW_PACK_POSITIVE",
    "_NSFW_POSITIVE",
    "_OBJECT_LAYOUT_NEGATIVE",
    "_OBJECT_LAYOUT_POSITIVE",
    "_OBJECTS_INTERACTION_POSITIVE",
    "_ONE_SHOT_NEGATIVE",
    "_ONE_SHOT_POSITIVE",
    "_PEOPLE_LAYOUT_NEGATIVE",
    "_PEOPLE_LAYOUT_POSITIVE",
    "_PENETRATION_POSITIVE",
    "_PERSPECTIVE_STABILITY_NEGATIVE",
    "_PERSPECTIVE_STABILITY_POSITIVE",
    "_POSE_NATURALNESS_NEGATIVE",
    "_POSE_NATURALNESS_POSITIVE",
    "_POSE_NEGATIVE",
    "_POSE_POSITIVE",
    "_QUALITY_PACK_NEGATIVE",
    "_QUALITY_PACK_POSITIVE",
    "_RELATIONSHIP_NEGATIVE",
    "_RELATIONSHIP_POSITIVE",
    "_SEX_POSITION_POSITIVE",
    "_SEX_VIEW_POSITIVE",
    "_SFW_CLOTHING_POSITIVE",
    "_SFW_ENVIRONMENT_POSITIVE",
    "_SFW_EXPRESSION_POSITIVE",
    "_SFW_MOOD_POSITIVE",
    "_SFW_NEGATIVE",
    "_SFW_POSE_POSITIVE",
    "_SFW_POSITIVE",
    "_SKIN_DETAIL_NEGATIVE",
    "_SKIN_DETAIL_POSITIVE",
    "_STYLE_LOCK_NEGATIVE",
    "_STYLE_LOCK_POSITIVE",
    "_STYLE_NEGATIVE",
    "_STYLE_POSITIVE",
    "_TYPOGRAPHY_NEGATIVE",
    "_TYPOGRAPHY_POSITIVE",
    "_UNWANTED_TEXT_NEGATIVE",
)

from typing import Dict, List, Tuple

from .content_control_tag_data import (
    conflicting_pairs_from_table,
    default_tag_data_dir,
    dict_pack,
    flat_pack,
    load_tag_tables,
)

_TAG_TABLES = load_tag_tables(default_tag_data_dir())
if not _TAG_TABLES:
    raise FileNotFoundError(
        f"Missing content-control tag CSVs under {default_tag_data_dir()}. "
        "Run: python -m scripts.tools dump_prompt_tag_csvs"
    )


def _dn(pack: str) -> Dict[str, List[str]]:
    d = dict_pack(_TAG_TABLES, pack)
    d.setdefault("none", [])
    return d


_BEST_MODEL_TAGS: List[str] = flat_pack(_TAG_TABLES, "best_model_tags")

_SFW_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "sfw_positive")
_SFW_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "sfw_negative")
_SFW_MOOD_POSITIVE: Dict[str, List[str]] = _dn("sfw_mood_positive")
_SFW_POSE_POSITIVE: Dict[str, List[str]] = _dn("sfw_pose_positive")
_SFW_CLOTHING_POSITIVE: Dict[str, List[str]] = _dn("sfw_clothing_positive")
_SFW_ENVIRONMENT_POSITIVE: Dict[str, List[str]] = _dn("sfw_environment_positive")
_SFW_EXPRESSION_POSITIVE: Dict[str, List[str]] = _dn("sfw_expression_positive")

_NSFW_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "nsfw_positive")
_NSFW_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "nsfw_negative")

_POSE_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "pose_positive")
_POSE_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "pose_negative_shared")
_ANGLE_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "angle_positive")
_ANGLE_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "angle_negative_shared")
_SEX_VIEW_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "sex_view_positive")
_DOMAIN_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "domain_positive")
_DOMAIN_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "domain_negative")
_BACKGROUND_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "background_positive")
_BACKGROUND_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "background_negative")
_PEOPLE_LAYOUT_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "people_layout_positive")
_PEOPLE_LAYOUT_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "people_layout_negative")
_RELATIONSHIP_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "relationship_positive")
_RELATIONSHIP_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "relationship_negative")
_OBJECT_LAYOUT_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "object_layout_positive")
_OBJECT_LAYOUT_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "object_layout_negative")
_HAND_POSITIVE: Dict[str, List[str]] = _dn("hand_positive")
_HAND_NEGATIVE: Dict[str, List[str]] = _dn("hand_negative")
_POSE_NATURALNESS_POSITIVE: Dict[str, List[str]] = _dn("pose_naturalness_positive")
_POSE_NATURALNESS_NEGATIVE: Dict[str, List[str]] = _dn("pose_naturalness_negative")
_TYPOGRAPHY_POSITIVE: Dict[str, List[str]] = _dn("typography_positive")
_TYPOGRAPHY_NEGATIVE: Dict[str, List[str]] = _dn("typography_negative")

_ONE_SHOT_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "one_shot_positive")
_ONE_SHOT_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "one_shot_negative")

_qp_sub = dict_pack(_TAG_TABLES, "quality_pack_positive")
_QUALITY_PACK_POSITIVE: Dict[str, List[str]] = {
    "none": [],
    "top": list(_BEST_MODEL_TAGS),
    "one_shot": list(_ONE_SHOT_POSITIVE) + list(_BEST_MODEL_TAGS),
    **_qp_sub,
}

_qn_sub = dict_pack(_TAG_TABLES, "quality_pack_negative")
_QUALITY_PACK_NEGATIVE: Dict[str, List[str]] = {
    "none": [],
    "top": list(flat_pack(_TAG_TABLES, "quality_pack_negative_top")),
    "one_shot": list(flat_pack(_TAG_TABLES, "quality_pack_negative_one_shot")),
    **_qn_sub,
}

_ADHERENCE_STANDARD_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "adherence_standard_positive")
_ADHERENCE_STRICT_EXTRA_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "adherence_strict_extra_positive")
_ADHERENCE_STANDARD_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "adherence_standard_negative")
_ADHERENCE_STRICT_EXTRA_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "adherence_strict_extra_negative")
_ADHERENCE_PACK_POSITIVE: Dict[str, List[str]] = {
    "none": [],
    "standard": list(_ADHERENCE_STANDARD_POSITIVE),
    "strict": list(_ADHERENCE_STANDARD_POSITIVE) + list(_ADHERENCE_STRICT_EXTRA_POSITIVE),
}
_ADHERENCE_PACK_NEGATIVE: Dict[str, List[str]] = {
    "standard": list(_ADHERENCE_STANDARD_NEGATIVE),
    "strict": list(_ADHERENCE_STANDARD_NEGATIVE) + list(_ADHERENCE_STRICT_EXTRA_NEGATIVE),
}

_LIGHTING_MODE_POSITIVE: Dict[str, List[str]] = _dn("lighting_mode_positive")
_LIGHTING_MODE_NEGATIVE: Dict[str, List[str]] = _dn("lighting_mode_negative")
_SKIN_DETAIL_POSITIVE: Dict[str, List[str]] = _dn("skin_detail_positive")
_SKIN_DETAIL_NEGATIVE: Dict[str, List[str]] = _dn("skin_detail_negative")
_NSFW_PACK_POSITIVE: Dict[str, List[str]] = _dn("nsfw_pack_positive")
_NSFW_PACK_NEGATIVE: Dict[str, List[str]] = _dn("nsfw_pack_negative")
_SEX_POSITION_POSITIVE: Dict[str, List[str]] = _dn("sex_position_positive")
_PENETRATION_POSITIVE: Dict[str, List[str]] = _dn("penetration_positive")
_BODY_PROPORTION_POSITIVE: Dict[str, List[str]] = _dn("body_proportion_positive")
_INTERACTION_INTENSITY: Dict[str, List[str]] = _dn("interaction_intensity_positive")
_CLOTHING_POSITIVE: Dict[str, List[str]] = _dn("clothing_positive")
_CLOTHING_NEGATIVE: Dict[str, List[str]] = _dn("clothing_negative")
_ADVANCED_POSE_POSITIVE: Dict[str, List[str]] = _dn("advanced_pose_positive")
_OBJECTS_INTERACTION_POSITIVE: Dict[str, List[str]] = _dn("objects_interaction_positive")
_ENVIRONMENT_DETAIL_POSITIVE: Dict[str, List[str]] = _dn("environment_detail_positive")
_STYLE_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "style_positive")
_STYLE_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "style_negative")
_STYLE_LOCK_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "style_lock_positive")
_STYLE_LOCK_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "style_lock_negative")
_UNWANTED_TEXT_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "unwanted_text_negative")
_COMPOSITION_POSITIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "composition_positive")
_COMPOSITION_NEGATIVE: Dict[str, List[str]] = dict_pack(_TAG_TABLES, "composition_negative")
_ARTIST_COMPOSITION_POSITIVE: Dict[str, List[str]] = _dn("artist_composition_positive")
_ARTIST_COMPOSITION_NEGATIVE: Dict[str, List[str]] = _dn("artist_composition_negative")
_PERSPECTIVE_STABILITY_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "perspective_stability_positive")
_PERSPECTIVE_STABILITY_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "perspective_stability_negative")
_DUPLICATE_SUBJECT_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "duplicate_subject_negative")
_CONFLICTING_TAG_PAIRS: List[Tuple[str, str]] = conflicting_pairs_from_table(_TAG_TABLES)

_ANTI_AI_LITE_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "anti_ai_lite_negative")
_ANTI_AI_STRONG_EXTRA_NEGATIVE: List[str] = flat_pack(_TAG_TABLES, "anti_ai_strong_extra_negative")
_ANTI_AI_STRONG_NEGATIVE: List[str] = list(_ANTI_AI_LITE_NEGATIVE) + list(_ANTI_AI_STRONG_EXTRA_NEGATIVE)
_ANTI_AI_LITE_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "anti_ai_lite_positive")
_ANTI_AI_STRONG_EXTRA_POSITIVE: List[str] = flat_pack(_TAG_TABLES, "anti_ai_strong_extra_positive")
_ANTI_AI_STRONG_POSITIVE: List[str] = list(_ANTI_AI_LITE_POSITIVE) + list(_ANTI_AI_STRONG_EXTRA_POSITIVE)

_HUMAN_MEDIA_POSITIVE: Dict[str, List[str]] = _dn("human_media_positive")
_HUMAN_MEDIA_NEGATIVE: Dict[str, List[str]] = _dn("human_media_negative")
_LORA_SCAFFOLD_POSITIVE: Dict[str, List[str]] = _dn("lora_scaffold_positive")
_LORA_SCAFFOLD_NEGATIVE: Dict[str, List[str]] = _dn("lora_scaffold_negative")
