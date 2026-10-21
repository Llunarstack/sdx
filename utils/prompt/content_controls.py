"""Content-control tag packs for harder generation targets.

``apply_content_controls`` appends the packs a mode selects (pose, view, domain, quality, adherence, anti-AI,
human media, LoRA scaffold, ...) to the positive and negative CSV prompts without repeats.
``infer_content_controls_from_prompt`` picks modes from the prompt's own words through ordered keyword
buckets; Danbooru-style comma prompts use tag-set matching for count tokens (1girl / solo / 2girls).
Tag text lives in ``content_control_tags_builtin.BUILTIN_TAG_TABLES`` as ``{pack: {mode: [tags]}}``
(mode ``"_"`` for flat packs).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from config.defaults._lexicon import dedupe_tags
from config.defaults.prompt_domains import (
    ANTI_AI_LOOK_NEGATIVE,
    ANTI_AI_LOOK_NEGATIVE_STRONG,
    NATURAL_LOOK_POSITIVE,
    NATURAL_LOOK_POSITIVE_DEEP,
)

from utils.prompt.tags import append_unique, split_tags

from .content_control_tags_builtin import BUILTIN_TAG_TABLES


def _flat(pack: str) -> list[str]:
    return list(BUILTIN_TAG_TABLES.get(pack, {}).get("_", []))


def _modes(pack: str) -> dict[str, list[str]]:
    return {mode: list(tags) for mode, tags in BUILTIN_TAG_TABLES.get(pack, {}).items()}


def _dn(pack: str) -> dict[str, list[str]]:
    """Mode pack whose "none" mode still merges (and so dedupes) the prompt."""
    d = _modes(pack)
    d.setdefault("none", [])
    return d


_BEST_MODEL_TAGS: list[str] = _flat("best_model_tags")

_POSE_POSITIVE: dict[str, list[str]] = _modes("pose_positive")
_POSE_NEGATIVE: list[str] = _flat("pose_negative_shared")
_ANGLE_POSITIVE: dict[str, list[str]] = _modes("angle_positive")
_ANGLE_NEGATIVE: list[str] = _flat("angle_negative_shared")
_SEX_VIEW_POSITIVE: dict[str, list[str]] = _modes("sex_view_positive")
_DOMAIN_POSITIVE: dict[str, list[str]] = _modes("domain_positive")
_DOMAIN_NEGATIVE: dict[str, list[str]] = _modes("domain_negative")
_BACKGROUND_POSITIVE: dict[str, list[str]] = _modes("background_positive")
_BACKGROUND_NEGATIVE: dict[str, list[str]] = _modes("background_negative")
_PEOPLE_LAYOUT_POSITIVE: dict[str, list[str]] = _modes("people_layout_positive")
_PEOPLE_LAYOUT_NEGATIVE: dict[str, list[str]] = _modes("people_layout_negative")
_RELATIONSHIP_POSITIVE: dict[str, list[str]] = _modes("relationship_positive")
_RELATIONSHIP_NEGATIVE: dict[str, list[str]] = _modes("relationship_negative")
_OBJECT_LAYOUT_POSITIVE: dict[str, list[str]] = _modes("object_layout_positive")
_OBJECT_LAYOUT_NEGATIVE: dict[str, list[str]] = _modes("object_layout_negative")
_HAND_POSITIVE: dict[str, list[str]] = _dn("hand_positive")
_HAND_NEGATIVE: dict[str, list[str]] = _dn("hand_negative")
_POSE_NATURALNESS_POSITIVE: dict[str, list[str]] = _dn("pose_naturalness_positive")
_POSE_NATURALNESS_NEGATIVE: dict[str, list[str]] = _dn("pose_naturalness_negative")
_TYPOGRAPHY_POSITIVE: dict[str, list[str]] = _dn("typography_positive")
_TYPOGRAPHY_NEGATIVE: dict[str, list[str]] = _dn("typography_negative")

_ONE_SHOT_POSITIVE: list[str] = _flat("one_shot_positive")
_ONE_SHOT_NEGATIVE: list[str] = _flat("one_shot_negative")

_QUALITY_PACK_POSITIVE: dict[str, list[str]] = {
    "none": [],
    "top": _BEST_MODEL_TAGS,
    "one_shot": _ONE_SHOT_POSITIVE + _BEST_MODEL_TAGS,
    **_modes("quality_pack_positive"),
}
_QUALITY_PACK_NEGATIVE: dict[str, list[str]] = {
    "none": [],
    "top": _flat("quality_pack_negative_top"),
    "one_shot": _flat("quality_pack_negative_one_shot"),
    **_modes("quality_pack_negative"),
}

_ADHERENCE_PACK_POSITIVE: dict[str, list[str]] = {
    "none": [],
    "standard": _flat("adherence_standard_positive"),
    "strict": _flat("adherence_standard_positive") + _flat("adherence_strict_extra_positive"),
}
_ADHERENCE_PACK_NEGATIVE: dict[str, list[str]] = {
    "standard": _flat("adherence_standard_negative"),
    "strict": _flat("adherence_standard_negative") + _flat("adherence_strict_extra_negative"),
}

_LIGHTING_MODE_POSITIVE: dict[str, list[str]] = _dn("lighting_mode_positive")
_LIGHTING_MODE_NEGATIVE: dict[str, list[str]] = _dn("lighting_mode_negative")
_SKIN_DETAIL_POSITIVE: dict[str, list[str]] = _dn("skin_detail_positive")
_SKIN_DETAIL_NEGATIVE: dict[str, list[str]] = _dn("skin_detail_negative")
_BODY_PROPORTION_POSITIVE: dict[str, list[str]] = _dn("body_proportion_positive")
_INTERACTION_INTENSITY: dict[str, list[str]] = _dn("interaction_intensity_positive")
_CLOTHING_POSITIVE: dict[str, list[str]] = _dn("clothing_positive")
_CLOTHING_NEGATIVE: dict[str, list[str]] = _dn("clothing_negative")
_ADVANCED_POSE_POSITIVE: dict[str, list[str]] = _dn("advanced_pose_positive")
_OBJECTS_INTERACTION_POSITIVE: dict[str, list[str]] = _dn("objects_interaction_positive")
_ENVIRONMENT_DETAIL_POSITIVE: dict[str, list[str]] = _dn("environment_detail_positive")
_STYLE_POSITIVE: dict[str, list[str]] = _modes("style_positive")
_STYLE_NEGATIVE: dict[str, list[str]] = _modes("style_negative")
_STYLE_LOCK_POSITIVE: list[str] = _flat("style_lock_positive")
_STYLE_LOCK_NEGATIVE: list[str] = _flat("style_lock_negative")
_UNWANTED_TEXT_NEGATIVE: list[str] = _flat("unwanted_text_negative")
_COMPOSITION_POSITIVE: dict[str, list[str]] = _modes("composition_positive")
_COMPOSITION_NEGATIVE: dict[str, list[str]] = _modes("composition_negative")
_ARTIST_COMPOSITION_POSITIVE: dict[str, list[str]] = _dn("artist_composition_positive")
_ARTIST_COMPOSITION_NEGATIVE: dict[str, list[str]] = _dn("artist_composition_negative")
_PERSPECTIVE_STABILITY_POSITIVE: list[str] = _flat("perspective_stability_positive")
_PERSPECTIVE_STABILITY_NEGATIVE: list[str] = _flat("perspective_stability_negative")
_DUPLICATE_SUBJECT_NEGATIVE: list[str] = _flat("duplicate_subject_negative")
_CONFLICTING_TAG_PAIRS: list[tuple[str, str]] = [
    (a.strip(), b.strip()) for a, _, b in (r.partition("|||") for r in _flat("conflicting_tag_pairs")) if b
]

_ANTI_AI_LITE_NEGATIVE: list[str] = dedupe_tags(ANTI_AI_LOOK_NEGATIVE.split(","))
_ANTI_AI_STRONG_NEGATIVE: list[str] = dedupe_tags(ANTI_AI_LOOK_NEGATIVE_STRONG.split(","))
_ANTI_AI_LITE_POSITIVE: list[str] = dedupe_tags(NATURAL_LOOK_POSITIVE.split(","))
_ANTI_AI_STRONG_POSITIVE: list[str] = dedupe_tags(NATURAL_LOOK_POSITIVE_DEEP.split(","))

_HUMAN_MEDIA_POSITIVE: dict[str, list[str]] = _dn("human_media_positive")
_HUMAN_MEDIA_NEGATIVE: dict[str, list[str]] = _dn("human_media_negative")
_LORA_SCAFFOLD_POSITIVE: dict[str, list[str]] = _dn("lora_scaffold_positive")
_LORA_SCAFFOLD_NEGATIVE: dict[str, list[str]] = _dn("lora_scaffold_negative")


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """Substring match on lowered prompt (same semantics as prior if/any chains)."""
    return any(n in haystack for n in needles)


def _first_bucket(haystack: str, buckets: Sequence[tuple[Sequence[str], str]]) -> str | None:
    """First matching keyword group wins (order preserves old elif chains)."""
    for keywords, value in buckets:
        if _contains_any(haystack, keywords):
            return value
    return None


def _merge_kv_pack(
    pos: str,
    neg: str,
    key: str,
    pos_map: dict[str, list[str]],
    neg_map: dict[str, list[str]] | None = None,
    *,
    shared_neg: Sequence[str] | None = None,
    pos_only: bool = False,
) -> tuple[str, str]:
    """Append ``pos_map[key]`` and the matching negatives (``shared_neg`` or ``neg_map[key]``); unknown keys skip."""
    if key not in pos_map:
        return pos, neg
    pos = append_unique(pos, pos_map[key])
    if pos_only:
        return pos, neg
    if shared_neg is not None:
        neg = append_unique(neg, shared_neg)
    elif neg_map is not None:
        neg = append_unique(neg, neg_map.get(key, []))
    return pos, neg


def _remove_conflicting_tags(text: str) -> str:
    """Drop the later tag of each contradictory pair."""
    tokens = split_tags(text)
    if not tokens:
        return text
    lowered = [t.lower() for t in tokens]
    to_remove = set()
    for a, b in _CONFLICTING_TAG_PAIRS:
        a_idx = next((i for i, t in enumerate(lowered) if a in t), None)
        b_idx = next((i for i, t in enumerate(lowered) if b in t), None)
        if a_idx is None or b_idx is None:
            continue
        if a_idx < b_idx:
            to_remove.add(b_idx)
        else:
            to_remove.add(a_idx)
    cleaned = [t for i, t in enumerate(tokens) if i not in to_remove]
    return ", ".join(cleaned)


# --- Keyword buckets for `infer_content_controls_from_prompt` (order = first-hit wins) ---

_INFER_SCENE_DOMAIN: tuple[tuple[tuple[str, ...], str], ...] = (
    (("car", "vehicle", "truck", "motorcycle", "bicycle", "bus"), "vehicles"),
    (("building", "architecture", "facade", "skyscraper", "interior"), "architecture"),
    (("object", "product shot", "tabletop", "still life"), "objects"),
)

_INFER_VIEW_ANGLE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("first person", "pov", "through the eyes"), "first_person"),
    (("bird eye", "bird's-eye", "overhead", "top down"), "bird_eye"),
    (("low angle", "from below", "worm eye", "worm's-eye"), "low_angle"),
)

# Golden hour / sunset are a time of day, not a rim setup; utils.prompt.art_direction plans that light.
_INFER_LIGHTING: tuple[tuple[tuple[str, ...], str], ...] = (
    (("studio lighting", "white background", "simple background"), "studio_softbox"),
    (("overcast", "cloudy day", "soft daylight"), "natural_daylight"),
)

_INFER_STYLE_MODE: tuple[tuple[tuple[str, ...], str], ...] = (
    (("3d render", "cg", "octane", "blender render", "unreal engine"), "3d"),
    (("photoreal", "raw photo", "dslr", "real photograph"), "photoreal"),
    (("semi realistic", "semi-realistic", "stylized realism", "2.5d"), "semi_real"),
    (("anime", "manga style", "cel shading", "visual novel"), "anime"),
)

_INFER_HUMAN_MEDIA_FILM: tuple[str, ...] = (
    "35mm",
    "kodak",
    "portra",
    "fuji film",
    "film photograph",
    "analog photo",
    "cinestill",
)
_INFER_HUMAN_MEDIA_DSLR: tuple[str, ...] = (
    "dslr",
    "shot on canon",
    "shot on nikon",
    "shot on sony",
    "85mm",
    "50mm lens",
    "full frame",
)
_INFER_HUMAN_MEDIA_PHOTO: tuple[str, ...] = (
    "photorealistic",
    "raw photo",
    "real photograph",
    "realistic photo",
    "iphone photo",
    "smartphone photo realistic",
)

_INFER_ADHERENCE_STANDARD_KW: tuple[str, ...] = (
    "as described",
    "faithful to prompt",
    "accurate to description",
    "per the prompt",
    "match the caption",
    "follow the prompt",
)
_INFER_ADHERENCE_STRICT_KW: tuple[str, ...] = (
    "strict adherence",
    "exactly as described",
    "match every detail",
    "no extra elements",
    "no additions beyond",
)


def infer_content_controls_from_prompt(prompt: str) -> dict[str, str]:
    """
    Infer likely control modes from prompt keywords.
    Returns partial kwargs for `apply_content_controls`.
    """
    raw_prompt = (prompt or "").strip()
    p = raw_prompt.lower()
    out: dict[str, str] = {}

    sd = _first_bucket(p, _INFER_SCENE_DOMAIN)
    if sd:
        out["scene_domain"] = sd

    va = _first_bucket(p, _INFER_VIEW_ANGLE)
    if va:
        out["view_angle"] = va

    if _contains_any(p, ("action pose", "acrobat", "parkour", "jumping", "spinning kick")):
        out["pose_mode"] = "action"

    if "lingerie" in p:
        out["clothing_mode"] = "lingerie"
    elif "bikini" in p:
        out["clothing_mode"] = "bikini"
    elif "torn" in p or "ripped clothes" in p:
        out["clothing_mode"] = "torn"

    tags = [x.strip().lower() for x in raw_prompt.split(",") if x.strip()]
    tagset = set(tags)
    comma_rich = "," in raw_prompt

    multi_noise = _contains_any(p, ("group", "crowd", "orgy")) or (
        bool(tagset & {"2girls", "2boys", "3girls"}) if comma_rich else _contains_any(p, ("2girls", "2boys", "3girls"))
    )
    if comma_rich:
        solo_hit = bool(tagset & {"1girl", "1boy", "1other", "solo"}) or _contains_any(p, ("single girl", "single boy"))
    else:
        solo_hit = _contains_any(p, ("1girl", "1boy", "1other", "single girl", "single boy", "solo"))

    if solo_hit and not multi_noise:
        out["composition_mode"] = "single_subject"
        out["people_layout"] = "solo"
    elif (comma_rich and bool(tagset & {"2girls", "2boys"})) or _contains_any(
        p, ("2girls", "2boys", "couple", "two girls", "two boys", "two people", "duo")
    ):
        out["people_layout"] = "duo"
        if out.get("composition_mode") in (None, "none"):
            out["composition_mode"] = "multi_character"

    if _contains_any(p, ("hand focus", "holding hands", "hands visible", "detailed hands", "fingers")):
        out["hand_mode"] = "stable"

    lm = _first_bucket(p, _INFER_LIGHTING)
    if lm:
        out["lighting_mode"] = lm

    sm = _first_bucket(p, _INFER_STYLE_MODE)
    if sm:
        out["style_mode"] = sm

    if _contains_any(p, _INFER_HUMAN_MEDIA_FILM):
        out["human_media_mode"] = "film"
    elif _contains_any(p, _INFER_HUMAN_MEDIA_DSLR):
        out["human_media_mode"] = "dslr"
    elif _contains_any(p, _INFER_HUMAN_MEDIA_PHOTO):
        out["human_media_mode"] = "photographic"

    parts = [x.strip() for x in raw_prompt.split(",") if x.strip()]
    n_parts = len(parts)
    plen = len(raw_prompt)
    inferred_adherence = "none"
    if n_parts >= 12 or plen >= 280:
        inferred_adherence = "standard"
    if n_parts >= 22 or plen >= 520:
        inferred_adherence = "strict"
    if _contains_any(p, _INFER_ADHERENCE_STANDARD_KW):
        inferred_adherence = "standard" if inferred_adherence == "none" else inferred_adherence
    if _contains_any(p, _INFER_ADHERENCE_STRICT_KW):
        inferred_adherence = "strict"
    if inferred_adherence != "none":
        out["adherence_pack"] = inferred_adherence

    if _contains_any(
        p,
        (
            "academic composition",
            "classical composition",
            "golden ratio",
            "golden spiral",
            "dynamic symmetry",
            "notan",
            "chiaroscuro composition",
            "arabesque",
        ),
    ):
        out["artist_composition"] = "classical"
    elif _contains_any(
        p,
        (
            "one point perspective",
            "1-point perspective",
            "two point perspective",
            "2-point perspective",
            "three point perspective",
            "3-point perspective",
            "vanishing point",
            "horizon line",
            "orthogonal perspective",
        ),
    ):
        out["artist_composition"] = "perspective"
    elif _contains_any(
        p,
        (
            "rule of thirds",
            "leading lines",
            "negative space",
            "visual weight",
            "focal plane",
        ),
    ):
        out["artist_composition"] = "standard"
    elif _contains_any(p, ("strong composition", "focal point", "visual hierarchy", "picture plane")):
        out["artist_composition"] = "lite"

    return out


def apply_content_controls(
    prompt: str,
    negative_prompt: str,
    *,
    pose_mode: str = "none",
    view_angle: str = "none",
    subject_sex: str = "none",
    scene_domain: str = "none",
    clothing_mode: str = "none",
    background_mode: str = "none",
    people_layout: str = "none",
    relationship_mode: str = "none",
    object_layout: str = "none",
    hand_mode: str = "none",
    pose_naturalness: str = "none",
    typography_mode: str = "none",
    quality_pack: str = "none",
    lighting_mode: str = "none",
    skin_detail_mode: str = "none",
    body_proportion: str = "none",
    interaction_intensity: str = "none",
    advanced_pose: str = "none",
    object_interaction: str = "none",
    environment_type: str = "none",
    style_mode: str = "none",
    style_lock: bool = False,
    anti_style_bleed: bool = False,
    composition_mode: str = "none",
    artist_composition: str = "none",
    anti_duplicate_subjects: bool = False,
    anti_perspective_drift: bool = False,
    cleanup_conflicting_tags: bool = False,
    allow_text_in_image: bool = False,
    one_shot_boost: bool = False,
    anti_ai_pack: str = "none",
    human_media_mode: str = "none",
    lora_scaffold: str = "none",
    adherence_pack: str = "none",
) -> tuple[str, str]:
    """
    Apply optional content controls to prompt + negative prompt.

    Returns `(updated_prompt, updated_negative_prompt)`.
    """
    p = (prompt or "").strip()
    n = (negative_prompt or "").strip()

    if one_shot_boost:
        p = append_unique(p, _ONE_SHOT_POSITIVE)
        n = append_unique(n, _ONE_SHOT_NEGATIVE)

    p, n = _merge_kv_pack(p, n, pose_mode, _POSE_POSITIVE, None, shared_neg=_POSE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, view_angle, _ANGLE_POSITIVE, None, shared_neg=_ANGLE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, subject_sex, _SEX_VIEW_POSITIVE, None, pos_only=True)

    p, n = _merge_kv_pack(p, n, scene_domain, _DOMAIN_POSITIVE, _DOMAIN_NEGATIVE)
    p, n = _merge_kv_pack(p, n, clothing_mode, _CLOTHING_POSITIVE, _CLOTHING_NEGATIVE)
    p, n = _merge_kv_pack(p, n, background_mode, _BACKGROUND_POSITIVE, _BACKGROUND_NEGATIVE)
    p, n = _merge_kv_pack(p, n, people_layout, _PEOPLE_LAYOUT_POSITIVE, _PEOPLE_LAYOUT_NEGATIVE)
    p, n = _merge_kv_pack(p, n, relationship_mode, _RELATIONSHIP_POSITIVE, _RELATIONSHIP_NEGATIVE)
    p, n = _merge_kv_pack(p, n, object_layout, _OBJECT_LAYOUT_POSITIVE, _OBJECT_LAYOUT_NEGATIVE)
    p, n = _merge_kv_pack(p, n, hand_mode, _HAND_POSITIVE, _HAND_NEGATIVE)
    p, n = _merge_kv_pack(p, n, pose_naturalness, _POSE_NATURALNESS_POSITIVE, _POSE_NATURALNESS_NEGATIVE)
    p, n = _merge_kv_pack(p, n, typography_mode, _TYPOGRAPHY_POSITIVE, _TYPOGRAPHY_NEGATIVE)
    p, n = _merge_kv_pack(p, n, quality_pack, _QUALITY_PACK_POSITIVE, _QUALITY_PACK_NEGATIVE)
    p, n = _merge_kv_pack(p, n, adherence_pack, _ADHERENCE_PACK_POSITIVE, _ADHERENCE_PACK_NEGATIVE)
    p, n = _merge_kv_pack(p, n, lighting_mode, _LIGHTING_MODE_POSITIVE, _LIGHTING_MODE_NEGATIVE)
    p, n = _merge_kv_pack(p, n, skin_detail_mode, _SKIN_DETAIL_POSITIVE, _SKIN_DETAIL_NEGATIVE)
    p, n = _merge_kv_pack(p, n, body_proportion, _BODY_PROPORTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, interaction_intensity, _INTERACTION_INTENSITY, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, advanced_pose, _ADVANCED_POSE_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, object_interaction, _OBJECTS_INTERACTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, environment_type, _ENVIRONMENT_DETAIL_POSITIVE, None, pos_only=True)

    p, n = _merge_kv_pack(p, n, style_mode, _STYLE_POSITIVE, _STYLE_NEGATIVE)

    if style_lock:
        p = append_unique(p, _STYLE_LOCK_POSITIVE)
        n = append_unique(n, _STYLE_LOCK_NEGATIVE)
    elif anti_style_bleed:
        n = append_unique(n, _STYLE_LOCK_NEGATIVE)

    p, n = _merge_kv_pack(p, n, composition_mode, _COMPOSITION_POSITIVE, _COMPOSITION_NEGATIVE)
    p, n = _merge_kv_pack(p, n, artist_composition, _ARTIST_COMPOSITION_POSITIVE, _ARTIST_COMPOSITION_NEGATIVE)

    if anti_duplicate_subjects:
        n = append_unique(n, _DUPLICATE_SUBJECT_NEGATIVE)

    if anti_perspective_drift:
        p = append_unique(p, _PERSPECTIVE_STABILITY_POSITIVE)
        n = append_unique(n, _PERSPECTIVE_STABILITY_NEGATIVE)

    if anti_ai_pack == "lite":
        p = append_unique(p, _ANTI_AI_LITE_POSITIVE)
        n = append_unique(n, _ANTI_AI_LITE_NEGATIVE)
    elif anti_ai_pack == "strong":
        p = append_unique(p, _ANTI_AI_STRONG_POSITIVE)
        n = append_unique(n, _ANTI_AI_STRONG_NEGATIVE)

    p, n = _merge_kv_pack(p, n, human_media_mode, _HUMAN_MEDIA_POSITIVE, _HUMAN_MEDIA_NEGATIVE)
    p, n = _merge_kv_pack(p, n, lora_scaffold, _LORA_SCAFFOLD_POSITIVE, _LORA_SCAFFOLD_NEGATIVE)

    if not allow_text_in_image:
        n = append_unique(n, _UNWANTED_TEXT_NEGATIVE)

    if cleanup_conflicting_tags:
        p = _remove_conflicting_tags(p)

    return p, n
