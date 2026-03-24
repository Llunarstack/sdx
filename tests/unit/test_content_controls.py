from pathlib import Path

from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt


def test_apply_content_controls_sfw_pose_view_domain():
    p, n = apply_content_controls(
        "portrait of a person",
        "blurry",
        safety_mode="sfw",
        pose_mode="complex",
        view_angle="low_angle",
        subject_sex="female",
        scene_domain="vehicles",
        allow_text_in_image=False,
    )
    assert "sfw" in p
    assert "dynamic full-body pose" in p
    assert "low-angle shot" in p
    assert "female anatomy consistency" in p
    assert "accurate vehicle geometry" in p
    assert "watermark" in n
    assert "deformed wheels" in n


def test_apply_content_controls_text_allowed_skips_text_suppression():
    p, n = apply_content_controls(
        "sign that says OPEN",
        "",
        view_angle="eye_level",
        allow_text_in_image=True,
    )
    assert "eye-level shot" in p
    assert "watermark" not in n
    assert "random text" not in n


def test_apply_content_controls_composition_and_perspective_guards():
    p, n = apply_content_controls(
        "two people in a street",
        "",
        composition_mode="group",
        anti_duplicate_subjects=True,
        anti_perspective_drift=True,
        cleanup_conflicting_tags=True,
    )
    assert "multi-character composition" in p
    assert "consistent perspective" in p
    assert "duplicate subject" in n
    assert "perspective drift" in n


def test_cleanup_conflicting_tags_keeps_first():
    p, _n = apply_content_controls(
        "portrait, front view, back view, daylight, night scene",
        "",
        cleanup_conflicting_tags=True,
    )
    lower = p.lower()
    assert "front view" in lower
    assert "back view" not in lower
    assert "daylight" in lower
    assert "night scene" not in lower


def test_infer_content_controls_from_prompt():
    inferred = infer_content_controls_from_prompt("first person POV of a sports car on city street")
    assert inferred.get("scene_domain") == "vehicles"
    assert inferred.get("view_angle") == "first_person"


def test_apply_content_controls_people_clothing_background_object_layout():
    p, n = apply_content_controls(
        "two characters in a city",
        "",
        clothing_mode="streetwear",
        background_mode="urban",
        people_layout="duo",
        relationship_mode="teamwork",
        object_layout="rule_of_thirds",
    )
    assert "streetwear style" in p
    assert "urban environment detail" in p
    assert "two people" in p
    assert "cooperative group action" in p
    assert "rule of thirds object placement" in p
    assert "third person artifact" in n


def test_style_mode_and_lock_for_3d_photoreal():
    p, n = apply_content_controls(
        "hero character portrait",
        "",
        style_mode="3d_photoreal",
        style_lock=True,
    )
    assert "photorealistic 3d render" in p
    assert "single coherent art direction" in p
    assert "style bleed" in n
    assert "toy-like plastic look" in n


def test_infer_style_mode_from_prompt():
    inferred = infer_content_controls_from_prompt("octane 3d render of a mech")
    assert inferred.get("style_mode") == "3d"


def test_infer_duo_prompt_sets_multi_character_composition():
    inf = infer_content_controls_from_prompt("2girls, cafe, different outfits")
    assert inf.get("people_layout") == "duo"
    assert inf.get("composition_mode") == "multi_character"


def test_apply_composition_multi_character_outfit_negatives():
    p, n = apply_content_controls("duo portrait", "", composition_mode="multi_character")
    assert "distinct clothing" in p
    assert "outfit swap" in n


def test_apply_content_controls_hand_and_typography_modes():
    p, n = apply_content_controls(
        "character poster",
        "",
        hand_mode="detailed",
        typography_mode="poster",
    )
    assert "detailed hands" in p
    assert "poster typography" in p
    assert "melted fingers" in n
    assert "broken headline" in n


def test_apply_content_controls_pose_naturalness():
    p, n = apply_content_controls(
        "two people posing",
        "",
        safety_mode="nsfw",
        pose_naturalness="intimate_natural",
    )
    assert "coherent interpersonal contact" in p
    assert "anatomically plausible paired pose" in p
    assert "impossible body contact" in n
    # Ensure this stacks with nsfw intent rather than replacing it.
    assert "adult content" in p


def test_apply_content_controls_quality_lighting_skin_and_nsfw_pack():
    p, n = apply_content_controls(
        "portrait shot",
        "",
        quality_pack="editorial",
        lighting_mode="studio_softbox",
        skin_detail_mode="natural_texture",
        nsfw_pack="soft",
    )
    assert "editorial photography look" in p
    assert "studio softbox lighting" in p
    assert "natural skin texture" in p
    assert "adult content" in p
    assert "plastic skin" in n or "waxy skin" in n


def test_quality_pack_top_adds_score_ladder_without_safety_mode():
    p, n = apply_content_controls(
        "1girl",
        "",
        safety_mode="none",
        quality_pack="top",
    )
    assert "score_9" in p
    assert "masterpiece" in p
    assert "worst quality" in n or "low quality" in n


def test_quality_pack_one_shot_and_one_shot_boost():
    p, n = apply_content_controls(
        "cat",
        "",
        safety_mode="none",
        quality_pack="one_shot",
        one_shot_boost=True,
    )
    assert "single subject focus" in p
    assert "score_9" in p or "masterpiece" in p
    assert "duplicate subject" in n


def test_sfw_positive_is_clean_no_explicit_tags():
    p, _n = apply_content_controls("portrait", "", safety_mode="sfw")
    lower = p.lower()
    assert "pussy" not in lower
    assert "creampie" not in lower
    assert "sfw" in lower


def test_infer_nsfw_sets_doggy_and_infer_solo_1girl():
    inf = infer_content_controls_from_prompt("1girl, solo, doggy style, explicit")
    assert inf.get("safety_mode") == "nsfw"
    assert inf.get("sex_position") == "doggy"
    assert inf.get("people_layout") == "solo"


def test_anti_ai_and_human_media_packs():
    p, n = apply_content_controls(
        "portrait",
        "",
        safety_mode="none",
        anti_ai_pack="lite",
        human_media_mode="film",
    )
    assert "natural skin texture" in p
    assert "plastic skin" in n
    assert "35mm film photograph" in p
    assert "digital noise pattern" in n


def test_lora_scaffold_blend():
    p, n = apply_content_controls(
        "1girl",
        "",
        lora_scaffold="blend",
    )
    assert "coherent lora fusion" in p
    assert "muddy lora blend" in n


def test_infer_human_media_from_film_keywords():
    inf = infer_content_controls_from_prompt("portrait, kodak portra, 35mm film")
    assert inf.get("human_media_mode") == "film"


def test_nsfw_uncensored_behavior():
    """Test that NSFW mode is now default and more comprehensive."""
    p, n = apply_content_controls(
        "beautiful woman nude explicit detail",
        "blurry",
        safety_mode="nsfw",
        nsfw_pack="explicit_detail",
    )
    assert "nsfw" in p.lower()
    assert "adult content" in p.lower()
    assert "detailed anatomy" in p.lower() or "high anatomy fidelity" in p.lower()
    assert "censorship bars" in n.lower() or "mosaic censor" in n.lower()
    assert "uncensored" in p.lower() or "explicit detail" in p.lower()


def test_nsfw_civitai_pack_wires_positives():
    p, n = apply_content_controls(
        "portrait",
        "",
        safety_mode="nsfw",
        nsfw_civitai_pack="action",
    )
    assert "action pose" in p
    assert "bad anatomy" in n


def test_nsfw_civitai_snippets_lite_short_tags():
    p, n = apply_content_controls(
        "portrait",
        "",
        safety_mode="nsfw",
        nsfw_civitai_pack="snippets_lite",
    )
    assert "zettai ryouiki" in p or "demon girl" in p
    assert "contradictory outfit tags" in n


def test_nsfw_civitai_hits_includes_csv_hot_tags():
    p, n = apply_content_controls(
        "portrait",
        "",
        safety_mode="nsfw",
        nsfw_civitai_pack="hits_lite",
    )
    assert "1girl" in p
    # First 48 of CIVITAI_HOT_TAGS follow merged CSV frequency (regenerated by curate script).
    assert "long hair" in p or "solo" in p
    assert "wrong eye color" in n


def test_civitai_frequency_trigger_bank(tmp_path):
    freq = tmp_path / "freq.txt"
    freq.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    p, _n = apply_content_controls(
        "",
        "",
        safety_mode="nsfw",
        civitai_trigger_bank="frequency_light",
        civitai_frequency_txt=str(freq),
    )
    assert "alpha" in p
    assert "beta" in p
    assert "gamma" in p


def test_civitai_trigger_bank_from_csv(tmp_path: Path):
    csv_path = tmp_path / "bank.csv"
    csv_path.write_text(
        "id,name,type,bases,triggers\n"
        "1,Test,LORA,Illustrious,alpha|beta\n"
        "2,Other,LORA,NoobAI,gamma\n",
        encoding="utf-8",
    )
    p, _n = apply_content_controls(
        "",
        "",
        safety_mode="nsfw",
        civitai_trigger_bank="light",
        civitai_model_bank_csv=str(csv_path),
    )
    assert "alpha" in p
    assert "beta" in p
    assert "gamma" in p


def test_extreme_sex_customization():
    """Test full customization for complex sexual scenes (e.g. extreme anatomy + specific positions)."""
    p, n = apply_content_controls(
        "2b and boy having sex",
        "bad anatomy",
        safety_mode="nsfw",
        sex_position="standing_missionary",
        penetration_detail="extreme",
        body_proportion="hyper",
        interaction_intensity="extreme",
        nsfw_pack="extreme",
    )
    assert "standing missionary position" in p.lower()
    assert "impossibly deep penetration" in p.lower() or "throat bulge" in p.lower()
    assert "hyper proportions" in p.lower() or "extreme" in p.lower()
    assert "brutal sex" in p.lower() or "maximum intensity" in p.lower()
    assert "censorship bars" in n.lower()


def test_adherence_pack_standard_and_strict():
    p, n = apply_content_controls("red dress, blue sky", "", adherence_pack="standard")
    assert "faithful to the prompt" in p
    assert "missing key elements" in n
    p2, n2 = apply_content_controls("portrait", "", adherence_pack="strict")
    assert "strict prompt adherence" in p2
    assert "hallucinated details" in n2


def test_quality_pack_micro_detail():
    p, n = apply_content_controls("still life", "", quality_pack="micro_detail")
    assert "texture fidelity" in p
    assert "mushy texture" in n


def test_infer_adherence_pack_from_long_prompt():
    long_tags = ", ".join(f"tag{i}" for i in range(15))
    inferred = infer_content_controls_from_prompt(long_tags)
    assert inferred.get("adherence_pack") == "standard"


def test_infer_adherence_pack_strict_keyword():
    inferred = infer_content_controls_from_prompt("portrait, exactly as described")
    assert inferred.get("adherence_pack") == "strict"
