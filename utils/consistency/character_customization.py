"""
Character profile utilities for prompt customization.

Supports both original and preexisting/canon character setups with optional
uncensored mode and adjustable identity strength.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _normalize_list_or_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _dedupe_keep_order(tokens: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        key = t.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(t.strip())
    return out


def build_character_prompt_additions(
    profile: Dict[str, Any],
    *,
    uncensored_mode: bool = False,
    character_strength: float = 1.0,
) -> Tuple[str, str]:
    """
    Build `(positive_additions, negative_additions)` from a character profile dict.
    """
    positive_tokens: List[str] = []
    negative_tokens: List[str] = []

    # Disambiguate multi-character / layout (optional JSON keys on character sheet)
    subject_label = str(profile.get("subject_label", "") or profile.get("character_slot", "")).strip()
    spatial_anchor = str(profile.get("spatial_anchor", "") or profile.get("screen_position", "")).strip()
    if subject_label:
        positive_tokens.append(f"subject label: {subject_label}")
    if spatial_anchor:
        positive_tokens.append(f"screen position: {spatial_anchor}")

    # Backward-compatible keys
    positive_tokens.extend(_normalize_list_or_str(profile.get("prompt")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("positive")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("appearance")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("style_tags")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("clothing")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("accessories")))
    negative_tokens.extend(_normalize_list_or_str(profile.get("negative")))
    negative_tokens.extend(_normalize_list_or_str(profile.get("negative_prompt")))

    # Rich character keys (original or existing/canon)
    positive_tokens.extend(_normalize_list_or_str(profile.get("character_name")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("source")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("franchise")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("aliases")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("identity_tokens")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("canon_traits")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("face_features")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("hair_features")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("body_features")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("wardrobe")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("materials")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("signature_items")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("pose_preferences")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("camera_preferences")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("environment_preferences")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("personality_cues")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("lighting_preferences")))
    positive_tokens.extend(_normalize_list_or_str(profile.get("artist_style_mix")))

    negative_tokens.extend(_normalize_list_or_str(profile.get("avoid_tokens")))
    negative_tokens.extend(_normalize_list_or_str(profile.get("identity_drift_negative")))

    # Gender presentation controls (opt-in metadata)
    gender_pres = str(profile.get("gender_presentation", "") or "").strip().lower()
    if gender_pres in {"androgynous", "androgynous presentation", "gender-ambiguous", "gender ambiguous"}:
        positive_tokens.append("androgynous presentation")
    elif gender_pres in {"male", "male-presenting", "man-presenting"}:
        positive_tokens.append("male-presenting")
    elif gender_pres in {"female", "female-presenting", "woman-presenting"}:
        positive_tokens.append("female-presenting")

    # Optional strict identity locks (for preexisting/canon stability)
    positive_tokens.extend(_normalize_list_or_str(profile.get("lock_tokens")))
    negative_tokens.extend(_normalize_list_or_str(profile.get("anti_lock_break_tokens")))

    # Legacy safety behavior becomes optional: only apply when not uncensored.
    if not uncensored_mode:
        lowered = [t.lower() for t in positive_tokens]
        if any("futa" in t or "trap" in t for t in lowered):
            positive_tokens = ["androgynous presentation" if ("futa" in t.lower() or "trap" in t.lower()) else t for t in positive_tokens]
            negative_tokens.append("explicit genital content")

    # Uncensored mode: remove anti-explicit blocks if present in provided negatives.
    if uncensored_mode:
        filtered_neg = []
        for t in negative_tokens:
            tl = t.lower()
            if any(x in tl for x in ("censorship bars", "mosaic censor", "explicit genital content", "sfw only")):
                continue
            filtered_neg.append(t)
        negative_tokens = filtered_neg

    positive_tokens = _dedupe_keep_order(positive_tokens)
    negative_tokens = _dedupe_keep_order(negative_tokens)

    # Identity strength: >1 repeats high-priority identity cues to improve adherence.
    strength = max(0.5, min(float(character_strength), 2.0))
    if strength > 1.0:
        identity_focus = _normalize_list_or_str(profile.get("identity_tokens")) + _normalize_list_or_str(
            profile.get("character_name")
        )
        identity_focus = _dedupe_keep_order(identity_focus)
        if identity_focus:
            for tok in identity_focus:
                positive_tokens.append(f"({tok})")

    return ", ".join(positive_tokens), ", ".join(negative_tokens)
