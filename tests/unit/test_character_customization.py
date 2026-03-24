from utils.consistency.character_customization import build_character_prompt_additions


def test_character_customization_supports_rich_profile_keys():
    profile = {
        "character_name": "Ayla Storm",
        "franchise": "original_character",
        "identity_tokens": ["short silver hair", "scar under left eye"],
        "canon_traits": ["stoic expression", "athletic build"],
        "wardrobe": ["black tactical jacket", "red armband"],
        "camera_preferences": ["three-quarter view"],
        "identity_drift_negative": ["wrong hair color", "missing scar"],
    }
    pos, neg = build_character_prompt_additions(profile, uncensored_mode=False, character_strength=1.0)
    assert "Ayla Storm" in pos
    assert "scar under left eye" in pos
    assert "three-quarter view" in pos
    assert "wrong hair color" in neg


def test_character_customization_uncensored_mode_keeps_explicit_terms():
    profile = {
        "positive": ["futa character design"],
        "negative": ["censorship bars", "sfw only"],
    }
    pos_safe, neg_safe = build_character_prompt_additions(profile, uncensored_mode=False, character_strength=1.0)
    assert "androgynous presentation" in pos_safe
    assert "explicit genital content" in neg_safe

    pos_unc, neg_unc = build_character_prompt_additions(profile, uncensored_mode=True, character_strength=1.0)
    assert "futa character design" in pos_unc
    assert "censorship bars" not in neg_unc


def test_character_customization_strength_reinforces_identity_tokens():
    profile = {
        "character_name": "Riven Hale",
        "identity_tokens": ["heterochromia", "freckled nose"],
    }
    pos, _neg = build_character_prompt_additions(profile, character_strength=1.8)
    assert "(heterochromia)" in pos
    assert "(Riven Hale)" in pos


def test_character_customization_spatial_and_subject_label():
    profile = {
        "subject_label": "left girl",
        "spatial_anchor": "left foreground",
        "wardrobe": ["yellow raincoat"],
    }
    pos, _neg = build_character_prompt_additions(profile, character_strength=1.0)
    assert "subject label: left girl" in pos
    assert "screen position: left foreground" in pos
    assert "yellow raincoat" in pos


def test_character_customization_supports_multi_character_merge_patterns():
    a = {
        "character_name": "Mira Vale",
        "identity_tokens": ["short red hair", "green eyes"],
        "lock_tokens": ["mira_signature_jacket"],
    }
    b = {
        "character_name": "Kade Orin",
        "identity_tokens": ["black undercut", "scar on chin"],
        "lock_tokens": ["kade_steel_gloves"],
    }
    pos_a, neg_a = build_character_prompt_additions(a, character_strength=1.0)
    pos_b, neg_b = build_character_prompt_additions(b, character_strength=1.0)
    merged_pos = ", ".join([pos_a, pos_b])
    merged_neg = ", ".join([neg_a, neg_b]).strip(", ")
    assert "Mira Vale" in merged_pos
    assert "Kade Orin" in merged_pos
    assert "mira_signature_jacket" in merged_pos
    assert "kade_steel_gloves" in merged_pos
    assert isinstance(merged_neg, str)
