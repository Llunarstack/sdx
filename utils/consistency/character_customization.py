"""
Character profile utilities for prompt customization.

Supports both original and preexisting/canon character setups with optional
uncensored mode and adjustable identity strength.
"""

from __future__ import annotations

from typing import Any

from config.defaults._lexicon import dedupe_tags

from utils.prompt.tags import str_list


def build_character_prompt_additions(
    profile: dict[str, Any],
    *,
    uncensored_mode: bool = False,
    character_strength: float = 1.0,
) -> tuple[str, str]:
    """
    Build `(positive_additions, negative_additions)` from a character profile dict.
    """
    positive_tokens: list[str] = []
    negative_tokens: list[str] = []

    # Disambiguate multi-character / layout (optional JSON keys on character sheet)
    subject_label = str(profile.get("subject_label", "") or profile.get("character_slot", "")).strip()
    spatial_anchor = str(profile.get("spatial_anchor", "") or profile.get("screen_position", "")).strip()
    if subject_label:
        positive_tokens.append(f"subject label: {subject_label}")
    if spatial_anchor:
        positive_tokens.append(f"screen position: {spatial_anchor}")

    # Backward-compatible keys
    positive_tokens.extend(str_list(profile.get("prompt")))
    positive_tokens.extend(str_list(profile.get("positive")))
    positive_tokens.extend(str_list(profile.get("appearance")))
    positive_tokens.extend(str_list(profile.get("style_tags")))
    positive_tokens.extend(str_list(profile.get("clothing")))
    positive_tokens.extend(str_list(profile.get("accessories")))
    negative_tokens.extend(str_list(profile.get("negative")))
    negative_tokens.extend(str_list(profile.get("negative_prompt")))

    # Rich character keys (original or existing/canon)
    positive_tokens.extend(str_list(profile.get("character_name")))
    positive_tokens.extend(str_list(profile.get("source")))
    positive_tokens.extend(str_list(profile.get("franchise")))
    positive_tokens.extend(str_list(profile.get("aliases")))
    positive_tokens.extend(str_list(profile.get("identity_tokens")))
    positive_tokens.extend(str_list(profile.get("canon_traits")))
    positive_tokens.extend(str_list(profile.get("face_features")))
    positive_tokens.extend(str_list(profile.get("hair_features")))
    positive_tokens.extend(str_list(profile.get("body_features")))
    positive_tokens.extend(str_list(profile.get("wardrobe")))
    positive_tokens.extend(str_list(profile.get("materials")))
    positive_tokens.extend(str_list(profile.get("signature_items")))
    positive_tokens.extend(str_list(profile.get("pose_preferences")))
    positive_tokens.extend(str_list(profile.get("camera_preferences")))
    positive_tokens.extend(str_list(profile.get("environment_preferences")))
    positive_tokens.extend(str_list(profile.get("personality_cues")))
    positive_tokens.extend(str_list(profile.get("lighting_preferences")))
    positive_tokens.extend(str_list(profile.get("artist_style_mix")))

    negative_tokens.extend(str_list(profile.get("avoid_tokens")))
    negative_tokens.extend(str_list(profile.get("identity_drift_negative")))

    # Gender presentation controls (opt-in metadata)
    gender_pres = str(profile.get("gender_presentation", "") or "").strip().lower()
    if gender_pres in {"androgynous", "androgynous presentation", "gender-ambiguous", "gender ambiguous"}:
        positive_tokens.append("androgynous presentation")
    elif gender_pres in {"male", "male-presenting", "man-presenting"}:
        positive_tokens.append("male-presenting")
    elif gender_pres in {"female", "female-presenting", "woman-presenting"}:
        positive_tokens.append("female-presenting")

    # Optional strict identity locks (for preexisting/canon stability)
    positive_tokens.extend(str_list(profile.get("lock_tokens")))
    negative_tokens.extend(str_list(profile.get("anti_lock_break_tokens")))

    # Legacy safety behavior becomes optional: only apply when not uncensored.
    if not uncensored_mode:
        lowered = [t.lower() for t in positive_tokens]
        if any("futa" in t or "trap" in t for t in lowered):
            positive_tokens = [
                "androgynous presentation" if ("futa" in t.lower() or "trap" in t.lower()) else t
                for t in positive_tokens
            ]
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

    positive_tokens = dedupe_tags(positive_tokens)
    negative_tokens = dedupe_tags(negative_tokens)

    # Identity strength: >1 repeats high-priority identity cues to improve adherence.
    strength = max(0.5, min(float(character_strength), 2.0))
    if strength > 1.0:
        identity_focus = str_list(profile.get("identity_tokens")) + str_list(profile.get("character_name"))
        identity_focus = dedupe_tags(identity_focus)
        if identity_focus:
            for tok in identity_focus:
                positive_tokens.append(f"({tok})")

    return ", ".join(positive_tokens), ", ".join(negative_tokens)
