"""
Prompt controls for harder generation targets.

``apply_content_controls`` merges optional tag packs into positive/negative CSV prompts using
shared helpers (dedupe, optional per-mode negatives, Civitai bank loaders).

``infer_content_controls_from_prompt`` maps free text to partial kwargs via ordered keyword
buckets; Danbooru-style comma prompts use tag-set matching for count tokens (1girl / solo / 2girls).

Tag text lives under ``data/prompt_tags/*.csv`` (see ``utils/prompt/content_control_tag_data.py``).
Regenerate from Python snapshots: ``python scripts/tools/dump_prompt_tag_csvs.py``.
"""
# ruff: noqa: F405

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .content_control_tags import *  # noqa: F401,F403,F405


def load_civitai_model_bank_triggers(
    csv_path: Optional[Path | str] = None,
    *,
    max_tokens: int = 120,
) -> List[str]:
    """
    Load unique trigger tokens from a Civitai model bank CSV (id,name,type,bases,triggers).

    Triggers column uses ``|``-separated tokens as written by ``fetch_civitai_nsfw_concepts.py``.
    Order is stable (first-seen wins). Empty / missing file returns [].
    """
    cap = max(0, int(max_tokens))
    if cap == 0:
        return []
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CIVITAI_MODEL_BANK_CSV
    if not path.is_file():
        return []
    out: List[str] = []
    seen: set[str] = set()
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = (row.get("triggers") or "").strip()
                if not raw:
                    continue
                for tok in raw.split("|"):
                    t = tok.strip()
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    out.append(t)
                    if len(out) >= cap:
                        return out
    except OSError:
        return []
    return out


def load_civitai_frequency_triggers(
    txt_path: Optional[Path | str] = None,
    *,
    max_tokens: int = 200,
) -> List[str]:
    """
    Load triggers from ``top_triggers_by_frequency.txt`` (one token per line, most common first).

    Regenerate that file with ``python scripts/tools/curate_civitai_triggers.py``.
    """
    cap = max(0, int(max_tokens))
    if cap == 0:
        return []
    path = Path(txt_path) if txt_path is not None else _DEFAULT_CIVITAI_FREQ_TXT
    if not path.is_file():
        return []
    out: List[str] = []
    seen: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= cap:
                break
    except OSError:
        return []
    return out


def _split_csv_tokens(text: str) -> List[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """Substring match on lowered prompt (same semantics as prior if/any chains)."""
    return any(n in haystack for n in needles)


def _first_bucket(haystack: str, buckets: Sequence[Tuple[Sequence[str], str]]) -> Optional[str]:
    """First matching keyword group wins (order preserves old elif chains)."""
    for keywords, value in buckets:
        if _contains_any(haystack, keywords):
            return value
    return None


def _merge_kv_pack(
    pos: str,
    neg: str,
    key: str,
    pos_map: Dict[str, List[str]],
    neg_map: Optional[Dict[str, List[str]]] = None,
    *,
    shared_neg: Optional[Sequence[str]] = None,
    pos_only: bool = False,
) -> Tuple[str, str]:
    """
    Append tokens for `key` from pos_map; optionally append negatives from shared_neg or neg_map[key].
    Skips unknown keys (not in pos_map). Empty pos lists are a no-op.
    """
    if key not in pos_map:
        return pos, neg
    pos = _append_unique_csv(pos, pos_map[key])
    if pos_only:
        return pos, neg
    if shared_neg is not None:
        neg = _append_unique_csv(neg, shared_neg)
    elif neg_map is not None:
        neg = _append_unique_csv(neg, neg_map.get(key, []))
    return pos, neg


def _append_unique_csv(base_text: str, additions: Sequence[str]) -> str:
    base = _split_csv_tokens(base_text)
    seen = {t.lower() for t in base}
    out = list(base)
    for token in additions:
        t = token.strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def _remove_conflicting_tags(text: str) -> str:
    """
    Remove obvious contradictory tag pairs by keeping the first occurrence in-order.
    """
    tokens = _split_csv_tokens(text)
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

_INFER_SCENE_DOMAIN: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("car", "vehicle", "truck", "motorcycle", "bicycle", "bus"), "vehicles"),
    (("building", "architecture", "facade", "skyscraper", "interior"), "architecture"),
    (("object", "product shot", "tabletop", "still life"), "objects"),
)

_INFER_VIEW_ANGLE: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("first person", "pov", "through the eyes"), "first_person"),
    (("bird eye", "bird's-eye", "overhead", "top down"), "bird_eye"),
    (("low angle", "from below", "worm eye", "worm's-eye"), "low_angle"),
)

_INFER_LIGHTING: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("golden hour", "sunset", "sunrise"), "dramatic_rim"),
    (("studio lighting", "white background", "simple background"), "studio_softbox"),
    (("overcast", "cloudy day", "soft daylight"), "natural_daylight"),
)

_INFER_STYLE_MODE: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("3d render", "cg", "octane", "blender render", "unreal engine"), "3d"),
    (("photoreal", "raw photo", "dslr", "real photograph"), "photoreal"),
    (("semi realistic", "semi-realistic", "stylized realism", "2.5d"), "semi_real"),
    (("anime", "manga style", "cel shading", "visual novel"), "anime"),
)

_INFER_HUMAN_MEDIA_FILM: Tuple[str, ...] = (
    "35mm",
    "kodak",
    "portra",
    "fuji film",
    "film photograph",
    "analog photo",
    "cinestill",
)
_INFER_HUMAN_MEDIA_DSLR: Tuple[str, ...] = (
    "dslr",
    "shot on canon",
    "shot on nikon",
    "shot on sony",
    "85mm",
    "50mm lens",
    "full frame",
)
_INFER_HUMAN_MEDIA_PHOTO: Tuple[str, ...] = (
    "photorealistic",
    "raw photo",
    "real photograph",
    "realistic photo",
    "iphone photo",
    "smartphone photo realistic",
)

_INFER_NSFW_TRIGGERS: Tuple[str, ...] = (
    "nsfw",
    "nude",
    "naked",
    "explicit",
    "adult content",
    "erotic",
    "uncensored",
    "porn",
    "hentai",
    "fucking",
    "fuck ",
    " sex ",
    "sex,",
    "sexual",
    "penetration",
    "ahegao",
    "creampie",
    "blowjob",
    "oral sex",
    " anal ",
    "vaginal",
    "dildo",
    "vibrator",
    "bukkake",
    "cum ",
    "cumming",
    "orgasm",
    "masturbat",
    "thighjob",
    "paizuri",
    "footjob",
    "spread legs",
    "doggy style",
    "missionary",
    "cowgirl",
    "reverse cowgirl",
    "nipples",
    "areola",
    "pussy",
    "penis",
    "cock ",
    "dick ",
    "cumshot",
    "facial cum",
)

_INFER_NSFW_PACK_EXTREME: Tuple[str, ...] = ("extreme", "macro", "throat bulge", "ahegao", "mind break", "brutal")
_INFER_NSFW_PACK_ROMANTIC: Tuple[str, ...] = ("romantic", "gentle", "tender", "couple", "cuddling")
_INFER_NSFW_PACK_DETAIL: Tuple[str, ...] = ("detailed", "closeup", "macro", "hi res", "highres")

_INFER_SFW_TRIGGERS: Tuple[str, ...] = (
    "sfw,",
    "sfw ",
    "safe for work",
    "family friendly",
    "pg-13",
    "pg rated",
    "wholesome only",
    "linkedin headshot",
    "corporate headshot",
    "christmas card",
    "school photo",
    "wedding photo",
    "toddler portrait",
)

_INFER_ADHERENCE_STANDARD_KW: Tuple[str, ...] = (
    "as described",
    "faithful to prompt",
    "accurate to description",
    "per the prompt",
    "match the caption",
    "follow the prompt",
)
_INFER_ADHERENCE_STRICT_KW: Tuple[str, ...] = (
    "strict adherence",
    "exactly as described",
    "match every detail",
    "no extra elements",
    "no additions beyond",
)

_WS_BOUNDARY = frozenset(" ,\n\t;|")


def _bounded_token(haystack_lower: str, needle: str) -> bool:
    """True if `needle` appears as a comma/whitespace-bounded token (reduces 'solo' in 'soloist', '69' in '1969')."""
    if not needle or not haystack_lower:
        return False
    n = len(needle)
    start = 0
    while True:
        i = haystack_lower.find(needle, start)
        if i < 0:
            return False
        left = haystack_lower[i - 1] if i > 0 else " "
        right = haystack_lower[i + n] if i + n < len(haystack_lower) else " "
        if left in _WS_BOUNDARY and right in _WS_BOUNDARY:
            return True
        start = i + max(1, n)


def _infer_nsfw_hit(p_lower: str) -> bool:
    if _bounded_token(p_lower, "69"):
        return True
    return _contains_any(p_lower, _INFER_NSFW_TRIGGERS)


def infer_content_controls_from_prompt(prompt: str) -> Dict[str, str]:
    """
    Infer likely control modes from prompt keywords.
    Returns partial kwargs for `apply_content_controls`.
    """
    raw_prompt = (prompt or "").strip()
    p = raw_prompt.lower()
    out: Dict[str, str] = {}

    sd = _first_bucket(p, _INFER_SCENE_DOMAIN)
    if sd:
        out["scene_domain"] = sd

    va = _first_bucket(p, _INFER_VIEW_ANGLE)
    if va:
        out["view_angle"] = va

    if _contains_any(p, ("action pose", "acrobat", "parkour", "jumping", "spinning kick")):
        out["pose_mode"] = "action"

    nsfw_hit = _infer_nsfw_hit(p)
    if nsfw_hit:
        out["safety_mode"] = "nsfw"
        if _contains_any(p, _INFER_NSFW_PACK_EXTREME):
            out["nsfw_pack"] = "extreme"
        elif _contains_any(p, _INFER_NSFW_PACK_ROMANTIC):
            out["nsfw_pack"] = "romantic"
        elif _contains_any(p, _INFER_NSFW_PACK_DETAIL):
            out["nsfw_pack"] = "explicit_detail"
        if "doggy" in p or "from behind" in p or "prone bone" in p:
            out["sex_position"] = "doggy"
        elif "reverse cowgirl" in p:
            out["sex_position"] = "cowgirl"
        elif "cowgirl" in p:
            out["sex_position"] = "cowgirl"
        elif "missionary" in p:
            out["sex_position"] = "missionary"
        elif "spoon" in p:
            out["sex_position"] = "spooning"
        elif "standing" in p and _contains_any(p, ("sex", "fuck", "penetration")):
            out["sex_position"] = "standing"
        if "lingerie" in p:
            out["clothing_mode"] = "lingerie"
        elif "bikini" in p:
            out["clothing_mode"] = "bikini"
        elif "torn" in p or "ripped clothes" in p:
            out["clothing_mode"] = "torn"
    elif _contains_any(p, _INFER_SFW_TRIGGERS):
        out["safety_mode"] = "sfw"

    tags = [x.strip().lower() for x in raw_prompt.split(",") if x.strip()]
    tagset = set(tags)
    comma_rich = "," in raw_prompt

    multi_noise = _contains_any(p, ("group", "crowd", "orgy")) or (
        bool(tagset & {"2girls", "2boys", "3girls"}) if comma_rich else _contains_any(p, ("2girls", "2boys", "3girls"))
    )
    if comma_rich:
        solo_hit = bool(tagset & {"1girl", "1boy", "1other", "solo"}) or _contains_any(
            p, ("single girl", "single boy")
        )
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

    return out


def apply_content_controls(
    prompt: str,
    negative_prompt: str,
    *,
    safety_mode: str = "none",
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
    nsfw_pack: str = "none",
    sex_position: str = "none",
    penetration_detail: str = "none",
    body_proportion: str = "none",
    interaction_intensity: str = "none",
    advanced_pose: str = "none",
    object_interaction: str = "none",
    environment_type: str = "none",
    sfw_mood: str = "none",
    sfw_pose: str = "none",
    sfw_clothing: str = "none",
    sfw_environment: str = "none",
    sfw_expression: str = "none",
    style_mode: str = "none",
    style_lock: bool = False,
    anti_style_bleed: bool = False,
    composition_mode: str = "none",
    anti_duplicate_subjects: bool = False,
    anti_perspective_drift: bool = False,
    cleanup_conflicting_tags: bool = False,
    allow_text_in_image: bool = False,
    nsfw_civitai_pack: str = "none",
    civitai_trigger_bank: str = "none",
    civitai_model_bank_csv: Optional[str] = None,
    civitai_frequency_txt: Optional[str] = None,
    one_shot_boost: bool = False,
    anti_ai_pack: str = "none",
    human_media_mode: str = "none",
    lora_scaffold: str = "none",
    adherence_pack: str = "none",
) -> Tuple[str, str]:
    """
    Apply optional content controls to prompt + negative prompt.

    Returns `(updated_prompt, updated_negative_prompt)`.
    """
    p = (prompt or "").strip()
    n = (negative_prompt or "").strip()

    if safety_mode == "sfw":
        p = _append_unique_csv(p, _SFW_POSITIVE)
        n = _append_unique_csv(n, _SFW_NEGATIVE)
    elif safety_mode == "nsfw":
        p = _append_unique_csv(p, _NSFW_POSITIVE)
        n = _append_unique_csv(n, _NSFW_NEGATIVE)

    if one_shot_boost:
        p = _append_unique_csv(p, _ONE_SHOT_POSITIVE)
        n = _append_unique_csv(n, _ONE_SHOT_NEGATIVE)

    p, n = _merge_kv_pack(p, n, sfw_mood, _SFW_MOOD_POSITIVE, None)
    p, n = _merge_kv_pack(p, n, sfw_pose, _SFW_POSE_POSITIVE, None)
    p, n = _merge_kv_pack(p, n, sfw_clothing, _SFW_CLOTHING_POSITIVE, None)
    p, n = _merge_kv_pack(p, n, sfw_environment, _SFW_ENVIRONMENT_POSITIVE, None)
    p, n = _merge_kv_pack(p, n, sfw_expression, _SFW_EXPRESSION_POSITIVE, None)

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
    p, n = _merge_kv_pack(p, n, nsfw_pack, _NSFW_PACK_POSITIVE, _NSFW_PACK_NEGATIVE)

    if safety_mode == "nsfw":
        p, n = _merge_kv_pack(p, n, nsfw_civitai_pack, _NSFW_CIVITAI_POSITIVE, _NSFW_CIVITAI_NEGATIVE)

    if safety_mode == "nsfw" and civitai_trigger_bank in _CIVITAI_TRIGGER_BANK_CAPS:
        cap = _CIVITAI_TRIGGER_BANK_CAPS[civitai_trigger_bank]
        if cap > 0:
            if civitai_trigger_bank.startswith("frequency_"):
                ft = (civitai_frequency_txt or "").strip()
                freq_path: Optional[Path] = Path(ft) if ft else None
                triggers = load_civitai_frequency_triggers(freq_path, max_tokens=cap)
            else:
                csv_opt = (civitai_model_bank_csv or "").strip()
                bank_path: Optional[Path] = Path(csv_opt) if csv_opt else None
                triggers = load_civitai_model_bank_triggers(bank_path, max_tokens=cap)
            if triggers:
                p = _append_unique_csv(p, triggers)

    p, n = _merge_kv_pack(p, n, sex_position, _SEX_POSITION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, penetration_detail, _PENETRATION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, body_proportion, _BODY_PROPORTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, interaction_intensity, _INTERACTION_INTENSITY, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, advanced_pose, _ADVANCED_POSE_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, object_interaction, _OBJECTS_INTERACTION_POSITIVE, None, pos_only=True)
    p, n = _merge_kv_pack(p, n, environment_type, _ENVIRONMENT_DETAIL_POSITIVE, None, pos_only=True)

    p, n = _merge_kv_pack(p, n, style_mode, _STYLE_POSITIVE, _STYLE_NEGATIVE)

    if style_lock:
        p = _append_unique_csv(p, _STYLE_LOCK_POSITIVE)
        n = _append_unique_csv(n, _STYLE_LOCK_NEGATIVE)
    elif anti_style_bleed:
        n = _append_unique_csv(n, _STYLE_LOCK_NEGATIVE)

    p, n = _merge_kv_pack(p, n, composition_mode, _COMPOSITION_POSITIVE, _COMPOSITION_NEGATIVE)

    if anti_duplicate_subjects:
        n = _append_unique_csv(n, _DUPLICATE_SUBJECT_NEGATIVE)

    if anti_perspective_drift:
        p = _append_unique_csv(p, _PERSPECTIVE_STABILITY_POSITIVE)
        n = _append_unique_csv(n, _PERSPECTIVE_STABILITY_NEGATIVE)

    if anti_ai_pack == "lite":
        p = _append_unique_csv(p, _ANTI_AI_LITE_POSITIVE)
        n = _append_unique_csv(n, _ANTI_AI_LITE_NEGATIVE)
    elif anti_ai_pack == "strong":
        p = _append_unique_csv(p, _ANTI_AI_STRONG_POSITIVE)
        n = _append_unique_csv(n, _ANTI_AI_STRONG_NEGATIVE)

    p, n = _merge_kv_pack(p, n, human_media_mode, _HUMAN_MEDIA_POSITIVE, _HUMAN_MEDIA_NEGATIVE)
    p, n = _merge_kv_pack(p, n, lora_scaffold, _LORA_SCAFFOLD_POSITIVE, _LORA_SCAFFOLD_NEGATIVE)

    if not allow_text_in_image:
        n = _append_unique_csv(n, _UNWANTED_TEXT_NEGATIVE)

    if cleanup_conflicting_tags:
        p = _remove_conflicting_tags(p)

    return p, n
