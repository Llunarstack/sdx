"""PromptParser extracts clothing, objects, people, and spatial relation tokens.

Also holds what used to be: test_prompt_adherence_parse, test_prompt_breakdown, test_prompt_budget, test_prompt_composer_artists, test_prompt_emphasis_import, test_prompt_lexicon_artist_helpers, test_prompt_research, test_prompt_stack, test_prompt_stack_exports, test_prompt_train_sample_parity, test_prompt_training_pkg_lazy, test_artist_composition_controls, test_auto_oc, test_composition_brief, test_creature_character_prompts, test_detailed_scene_entities, test_multi_instance_scene, test_photo_realism, test_physics_material_prompts, test_prompt_tags, test_rag_prompt_gen_searcher.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from config.prompts import HUMANOID_MONSTER_PROMPT_KEYWORDS
from models.conditioning import PromptParser, attribute_binding_heatmap
from pipelines.book_comic import prompt_lexicon
from utils.captioning import PromptResearchResult, _seed_from_description, research_prompt_for_image, retrieve_rag_facts
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_core import apply_best_model_policy
from utils.prompt import rag as rp
from utils.prompt.art_direction import (
    apply_art_direction_to_args,
    apply_composition_brief,
    composition_brief_warranted,
    infer_photo_realism_controls,
    is_photographic_prompt,
    photo_realism_fragments,
    recommend_photo_post_profile,
)
from utils.prompt.content_controls import apply_content_controls, infer_content_controls_from_prompt
from utils.prompt.scene import (
    apply_detailed_scene_boost,
    apply_multi_instance_preset,
    describe_limitation_short,
    detailed_scene_warrants_boost,
    extract_key_segments,
    infer_auto_original_character,
    multi_instance_auto_settings,
    print_multi_instance_hints,
    prompt_requests_original_character,
    suggest_creature_prompt_addons,
    suggest_physics_prompt_addons,
)
from utils.prompt.stack import (
    CLAUSE_REGISTRY,
    PromptContext,
    StackMode,
    analyze_prompt,
    append_unique,
    apply_clauses,
    apply_sample_prompt_stack,
    apply_training_guidance_pair,
    fit_tags,
    list_clauses,
    merge_guidance_for_training_caption,
    resolve_content_controls,
    run_prompt_stack,
    split_tags,
    tag_token_counts,
)
from utils.prompt.text import (
    ArtistRegistry,
    apply_prompt_breakdown,
    build_breakdown,
    compose_prompt,
    expand_artist_mentions,
    filter_negative_by_positive,
    merge_csv_unique,
    normalize_artist_tag,
    parse_blocks,
    parse_prompt_emphasis,
    token_weights_from_cleaned_segments,
    warrant_prompt_breakdown,
)

# ==================================================================================================
# test_prompt_adherence_parse: PromptParser extracts clothing, objects, people, and spatial relation tokens.
# ==================================================================================================


def test_parse_binds_red_dress_and_blue_suit():
    parsed = PromptParser().parse("a woman in a red dress and a man in a blue suit")
    subjects = {t.subject for t in parsed.triples}
    assert "dress" in subjects
    assert "suit" in subjects
    dress = next(t for t in parsed.triples if t.subject == "dress")
    suit = next(t for t in parsed.triples if t.subject == "suit")
    assert "red" in dress.attributes
    assert "blue" in suit.attributes


def test_parse_spatial_left_of_and_colored_objects():
    parsed = PromptParser().parse("a red cube to the left of a blue sphere")
    subjects = {t.subject for t in parsed.triples}
    assert "cube" in subjects
    assert "sphere" in subjects
    assert any("left" in t.relations for t in parsed.triples)
    cube = next(t for t in parsed.triples if t.subject == "cube")
    assert "red" in cube.attributes


def test_binding_heatmap_reads_parsed_subject_tokens():
    parsed = PromptParser().parse("a red dress")
    attn = torch.zeros(1, 4, 8)
    attn[0, 0, 1:3] = 1.0  # patch 0 attends to "red dress" (words 1-2)
    heat = attribute_binding_heatmap(attn, parsed, spatial_h=2, spatial_w=2)
    assert heat.flatten().tolist() == [1.0, 0.0, 0.0, 0.0]


# ==================================================================================================
# test_prompt_breakdown: Tests for heuristic prompt breakdown / section reordering.
# ==================================================================================================


def test_warrant_long():
    assert warrant_prompt_breakdown("x" * 120)


def test_warrant_many_clauses():
    assert warrant_prompt_breakdown("a, b, c, d, e")


def test_warrant_short_no():
    assert not warrant_prompt_breakdown("a red apple")
    assert not warrant_prompt_breakdown("")


def test_subject_first_moves_quality_after_subject():
    raw = "masterpiece, 8k, a knight in armor, golden hour"
    res = build_breakdown(raw, order="subject_first")
    assert res.ordered_flat.startswith("a knight in armor")
    assert "masterpiece" in res.ordered_flat
    assert "golden hour" in res.ordered_flat


def test_quality_first_leads_with_quality_tags():
    raw = "a knight in armor, masterpiece, 8k"
    res = build_breakdown(raw, order="quality_first")
    assert res.ordered_flat.startswith("masterpiece")


def test_labeled_format_has_section_labels():
    raw = "a fox in a forest, cinematic lighting, anime style"
    flat, labeled = apply_prompt_breakdown(raw, order="subject_first", output_format="labeled")
    assert flat
    assert "SUBJECTS:" in labeled or "ENVIRONMENT:" in labeled
    assert labeled.count("\n") >= 1


def test_ordered_format_identical_strings():
    raw = "wide angle, a castle, oil painting"
    a, b = apply_prompt_breakdown(raw, output_format="ordered")
    assert a == b


# ==================================================================================================
# test_prompt_budget: Prompt flooding: token budget, medium/subject gating of default packs, and the bugs that caused it.
# ==================================================================================================


def _default_run(prompt: str, *extra: str) -> tuple[str, str]:
    """sample.py's default prompt path (parser defaults + quality policy + stack), no model."""
    args = build_sample_parser().parse_args(["--ckpt", "x.pt", "--prompt", prompt, *extra])
    args._user_prompt, args._user_negative = prompt, str(args.negative_prompt or "")
    apply_art_direction_to_args(args)
    apply_best_model_policy(args, prompt=prompt)
    return apply_sample_prompt_stack(args, args.prompt)


def _tokens(text: str) -> int:
    return sum(tag_token_counts([t.strip() for t in text.split(",") if t.strip()])) + text.count(",") + 1


def test_landscape_painting_is_not_flooded_with_people_or_photo_packs():
    pos, neg = _default_run("misty mountain lake at dawn, oil painting")
    assert pos.startswith("misty mountain lake at dawn, oil painting, ")
    assert "horizon on the lower third" in pos  # art direction follows the user's words
    for word in ("skin", "hands", "fingers", "eyes", "pixel", "vector", "dslr", "bokeh", "lens", "photo"):
        assert not re.search(rf"\b{word}\b", pos, re.I), word
    assert _tokens(pos) <= _tokens("misty mountain lake at dawn, oil painting") + 96
    assert "bad hands" not in neg and "low quality" in neg


def test_pixel_art_is_not_style_fusion_or_photoreal():
    pos, _ = _default_run("pixel art knight reflected in a puddle")
    for phrase in ("photorealistic anime", "painterly photograph", "pores", "bokeh", "sensor"):
        assert phrase not in pos, phrase


def test_user_words_always_survive_and_come_first():
    long_prompt = ", ".join(f"detail number {i} of the ancient harbour town" for i in range(60))
    pos, _ = _default_run(long_prompt)
    assert pos.startswith(long_prompt)
    pos, neg = _default_run("a red fox", "--prompt-budget", "0", "--negative-prompt", "people, text")
    assert pos == "a red fox"
    assert neg.startswith("people, text")


def test_fit_tags_protects_dedupes_and_respects_both_limits():
    kept, dropped = fit_tags(["a", "b"], ["B", "c", "d e f g"], limit=300, extra=3)
    assert kept == ["a", "b", "c"] and dropped == 1
    huge = [f"word{i}" for i in range(50)]
    kept, _ = fit_tags(huge, ["extra"], limit=10, extra=-1)
    assert kept == huge


def test_negative_filter_drops_whole_tags_against_requested_words():
    out = filter_negative_by_positive(
        "woman with blurry background, holding hands", "blurry, plastic skin, bad hands, text"
    )
    assert out.split(", ") == ["plastic skin", "bad hands", "text"]
    assert filter_negative_by_positive("low angle shot of a tower", "low quality") == "low quality"


def test_category_and_lighting_bugs():
    from config.styles import guidance_fragments
    from utils.prompt.content_controls import infer_content_controls_from_prompt
    from utils.prompt.special_prompt_helpers import classify_prompt_category

    assert classify_prompt_category("pixel art knight reflected in a puddle") != "style_fusion"
    assert classify_prompt_category("low-key noir portrait, black and white") != "style_fusion"
    assert classify_prompt_category("photorealistic anime girl") == "style_fusion"
    assert infer_content_controls_from_prompt("beach at golden hour").get("lighting_mode") != "dramatic_rim"
    assert guidance_fragments("misty mountain lake", "none", include_photography=True, anatomy_mode="strong") == (
        "",
        "",
    )
    assert (
        "accurate anatomy"
        in guidance_fragments("a woman dancing", "none", include_photography=True, anatomy_mode="strong")[0]
    )


# ==================================================================================================
# test_prompt_composer_artists: Tests for @artist resolution and +category prompt composer.
# ==================================================================================================


def test_normalize_artist_tag():
    assert normalize_artist_tag("Kantoku") == "kantoku"
    assert normalize_artist_tag("hiten_(hitenkei)") == "hiten (hitenkei)"


def test_registry_resolves_aliases():
    reg = ArtistRegistry()
    reg.add("kantoku", count=100)
    reg.add("wlop", count=50)
    assert reg.resolve("Kantoku") == "kantoku"
    assert reg.resolve("WLOP") == "wlop"
    assert reg.resolve("unknown_artist_xyz") == "unknown artist xyz"


def test_expand_any_artist():
    reg = ArtistRegistry()
    reg.add("some_pixiv_name", count=1)
    out, artists = expand_artist_mentions("@Some_Pixiv_Name", registry=reg)
    assert "some pixiv name" in out
    assert artists == ["some pixiv name"]


def test_parse_blocks():
    base, blocks = parse_blocks(
        "@wlop sunset +character: 1girl, blue hair +building: art deco tower +car: red sports car"
    )
    assert "@" not in base or "wlop" in base.lower()
    assert "character" in blocks
    assert "building" in blocks
    assert "car" in blocks


def test_compose_full():
    cp = compose_prompt(
        "@wlop +character: 1girl, silver hair +vehicle: red sports car, low angle",
        artist_registry=ArtistRegistry(),
    )
    assert "wlop" in cp.positive
    assert "1girl" in cp.positive
    assert "silver hair" in cp.positive
    assert "motor vehicle" in cp.positive or "vehicle" in cp.positive
    assert "red sports car" in cp.positive


# ==================================================================================================
# test_prompt_emphasis_import: Regression: prompt_emphasis must import when torch is absent (smoke_imports path).
# ==================================================================================================

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_emphasis_import_without_torch():
    code = """
import sys
sys.modules['torch'] = None
from utils.prompt.text import parse_prompt_emphasis
cleaned, _ = parse_prompt_emphasis('(hi) there')
assert cleaned == 'hi there', cleaned
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)


# ==================================================================================================
# test_prompt_lexicon_artist_helpers: Tests for artist-facing helpers in pipelines.book_comic.prompt_lexicon.
# ==================================================================================================


def test_artist_craft_bundle_includes_selected_fragments():
    out = prompt_lexicon.artist_craft_bundle(
        craft_profile="manga_pro",
        shot_language="manga_dynamic",
        pacing_plan="decompressed",
        lettering_craft="strict",
        value_plan="bw_hierarchy",
        screentone_plan="dramatic",
    )
    assert "right-to-left panel flow" in out
    assert "dynamic manga framing" in out
    assert "decompressed pacing" in out
    assert "strict lettering discipline" in out
    assert "black-white value hierarchy" in out
    assert "dramatic screentone contrast" in out


def test_suggest_negative_addon_with_artist_lettering_strict():
    out = prompt_lexicon.suggest_negative_addon(
        use_lexicon_negative=True,
        user_negative="blurry text",
        production_tier=False,
        artist_lettering_strict=True,
    )
    assert "blurry text" in out
    assert "crossing balloon tails" in out


def test_original_character_bundle_contains_identity_anchors():
    out = prompt_lexicon.original_character_bundle(
        name="Astra Vale",
        archetype="space_pilot",
        visual_traits="silver undercut, amber eyes, cheek scar",
        wardrobe="flight jacket with orange stripes",
        silhouette="triangular jacket shoulders, slim lower silhouette",
        color_motifs="teal and orange",
        expression_sheet="confident smirk, focused glare",
    )
    assert "original character Astra Vale" in out
    assert "functional sci-fi costume logic" in out
    assert "signature traits" in out
    assert "consistent wardrobe" in out


def test_resolve_artist_controls_pack_and_override():
    out = prompt_lexicon.resolve_artist_controls(
        artist_pack="manga_cinematic",
        pacing_plan="compressed",
    )
    assert out["craft_profile"] == "manga_pro"
    assert out["shot_language"] == "manga_dynamic"
    assert out["pacing_plan"] == "compressed"  # explicit override wins


def test_artist_craft_bundle_includes_artist_style_profile():
    out = prompt_lexicon.artist_craft_bundle(
        craft_profile="manga_pro",
        artist_style_profile="shinkai_cinematic_anime",
    )
    assert "Makoto Shinkai inspired anime language" in out


def test_resolve_artist_controls_new_game_pack():
    out = prompt_lexicon.resolve_artist_controls(
        artist_pack="game_riot_splash",
    )
    assert out["artist_style_profile"] == "riot_splash_fantasy"
    assert out["shot_language"] == "cinematic"


def test_resolve_oc_controls_pack_and_override():
    out = prompt_lexicon.resolve_oc_controls(
        oc_pack="heroine_scifi",
        name="Nia Star",
        wardrobe="pilot suit with white stripes",
    )
    assert out["name"] == "Nia Star"
    assert out["archetype"] == "space_pilot"
    assert out["wardrobe"] == "pilot suit with white stripes"  # explicit override wins


def test_resolve_book_style_controls_pack_and_override():
    out = prompt_lexicon.resolve_book_style_controls(
        book_style_pack="webtoon_nsfw_romance",
        artist_pack="manga_cinematic",
    )
    assert out["artist_pack"] == "manga_cinematic"  # explicit override wins
    assert out["oc_pack"] == "none"


def test_resolve_humanize_controls_pack_and_override():
    out = prompt_lexicon.resolve_humanize_controls(
        humanize_pack="painterly",
        asymmetry_level="strong",
    )
    assert out["humanize_profile"] == "painterly"
    assert out["materiality_mode"] == "canvas"
    assert out["asymmetry_level"] == "strong"


def test_humanize_prompt_and_negative_helpers():
    p = prompt_lexicon.humanize_prompt_bundle(
        humanize_profile="balanced",
        imperfection_level="balanced",
        materiality_mode="paper",
        asymmetry_level="lite",
    )
    n = prompt_lexicon.humanize_negative_addon("balanced")
    assert "human-made mark-making cadence" in p
    assert "paper tooth" in p
    assert "ai soup texture" in n


def test_infer_auto_humanize_controls_storyboard_and_seinen():
    sb = prompt_lexicon.infer_auto_humanize_controls(
        book_type="storyboard",
        lexicon_style="none",
    )
    assert sb["humanize_profile"] == "lite"
    ns = prompt_lexicon.infer_auto_humanize_controls(
        book_type="manga",
        lexicon_style="seinen",
    )
    assert ns["humanize_profile"] == "painterly"


def test_resolve_art_medium_controls_pack_and_override():
    out = prompt_lexicon.resolve_art_medium_controls(
        art_medium_pack="anime_3d_pro",
        art_medium_variant="web_comic",
    )
    assert out["family"] == "anime_cartoon_webcomic"
    assert out["variant"] == "web_comic"  # explicit override wins


def test_art_medium_bundle_supports_requested_families():
    digital = prompt_lexicon.art_medium_bundle(
        family="digital_art",
        variant="painting",
    )
    drawing = prompt_lexicon.art_medium_bundle(
        family="drawing_art",
        variant="ink",
    )
    p3d = prompt_lexicon.art_medium_bundle(
        family="digital_3d_art",
        variant="stylized",
    )
    painting = prompt_lexicon.art_medium_bundle(
        family="painting_art",
        variant="watercolor",
    )
    realistic = prompt_lexicon.art_medium_bundle(
        family="realistic_art",
        variant="photoreal",
    )
    anime_web = prompt_lexicon.art_medium_bundle(
        family="anime_cartoon_webcomic",
        variant="web_comic",
    )
    assert "digital painting workflow" in digital
    assert "ink drawing medium" in drawing
    assert "stylized 3d art medium" in p3d
    assert "watercolor painting medium" in painting
    assert "photoreal medium treatment" in realistic
    assert "web comic medium" in anime_web


def test_art_medium_bundle_supports_mixed_media_family():
    mixed = prompt_lexicon.art_medium_bundle(
        family="mixed_media_art",
        variant="risograph",
    )
    assert "risograph print medium" in mixed


def test_art_medium_bundle_supports_new_digital_3d_drawing_painting_variants():
    d = prompt_lexicon.art_medium_bundle(family="digital_art", variant="splash_art")
    g3 = prompt_lexicon.art_medium_bundle(family="digital_3d_art", variant="unreal_cinematic")
    dr = prompt_lexicon.art_medium_bundle(family="drawing_art", variant="lineart_sheet")
    p = prompt_lexicon.art_medium_bundle(family="painting_art", variant="watercolor_botanical")
    assert "digital splash-art medium" in d
    assert "unreal-cinematic 3d medium" in g3
    assert "lineart character-sheet medium" in dr
    assert "watercolor botanical medium" in p


def test_style_snippet_supports_extended_style_keys():
    assert "cyberpunk art direction" in prompt_lexicon.style_snippet("cyberpunk")
    assert "ukiyo-e inspired treatment" in prompt_lexicon.style_snippet("ukiyo_e")
    assert "octane-like render style" in prompt_lexicon.style_snippet("render_octane")


def test_style_snippet_supports_huge_additional_keys():
    assert "baroque-inspired visual language" in prompt_lexicon.style_snippet("baroque")
    assert "shonen battle anime style" in prompt_lexicon.style_snippet("anime_shonen_battle")
    assert "archviz realism style" in prompt_lexicon.style_snippet("archviz_real")
    assert "anime-game toon-PBR hybrid" in prompt_lexicon.style_snippet("anime_game_toon_pbr")
    assert "honkai star rail inspired 3d anime style" in prompt_lexicon.style_snippet("honkai_starrail_3d_anime")
    assert "digital fantasy splash-art style" in prompt_lexicon.style_snippet("digital_fantasy_splash")
    assert "cinematic PBR key-art style" in prompt_lexicon.style_snippet("pbr_cinematic_keyart")
    assert "character-sheet lineart style" in prompt_lexicon.style_snippet("lineart_character_sheet")
    assert "classical oil portrait style" in prompt_lexicon.style_snippet("oil_portrait_classical")


def test_resolve_art_medium_controls_with_new_pack():
    out = prompt_lexicon.resolve_art_medium_controls(
        art_medium_pack="sports_photo_pro",
    )
    assert out["family"] == "realistic_art"
    assert out["variant"] == "sports_photo"


def test_resolve_art_medium_controls_with_popular_new_packs():
    d = prompt_lexicon.resolve_art_medium_controls(art_medium_pack="digital_splash_master")
    g3 = prompt_lexicon.resolve_art_medium_controls(art_medium_pack="anime_game_3d_pro")
    dr = prompt_lexicon.resolve_art_medium_controls(art_medium_pack="lineart_sheet_pro")
    p = prompt_lexicon.resolve_art_medium_controls(art_medium_pack="gouache_poster_master")
    assert d["family"] == "digital_art" and d["variant"] == "splash_art"
    assert g3["family"] == "digital_3d_art" and g3["variant"] == "anime_game_3d"
    assert dr["family"] == "drawing_art" and dr["variant"] == "lineart_sheet"
    assert p["family"] == "painting_art" and p["variant"] == "gouache_poster"


def test_resolve_color_render_controls_pack_and_override():
    out = prompt_lexicon.resolve_color_render_controls(
        color_render_pack="pbr_3d_master",
        shading_technique="subsurface",
    )
    assert out["color_theory_mode"] == "split_complementary"
    assert out["render_pipeline"] == "pbr_3d"
    assert out["shading_technique"] == "subsurface"


def test_color_render_bundle_contains_expected_fragments():
    out = prompt_lexicon.color_render_bundle(
        color_theory_mode="triadic",
        gradient_blend_mode="toon_steps",
        shading_technique="cel",
        render_pipeline="anime_2d",
    )
    assert "triadic palette balance" in out
    assert "toon-step gradients" in out
    assert "cel-shading logic" in out
    assert "anime 2d render discipline" in out


def test_resolve_artist_controls_new_anime3d_pack():
    out = prompt_lexicon.resolve_artist_controls(
        artist_pack="anime3d_hsr_cinematic",
    )
    assert out["artist_style_profile"] == "alpha_3d_anime_keyart"
    assert out["shot_language"] == "cinematic"


def test_suggest_popular_style_presets_for_anime_3d_query():
    rows = prompt_lexicon.suggest_popular_style_presets(
        "anime 3d game keyart",
        category="3d",
        limit=5,
    )
    assert rows
    ids = {r["id"] for r in rows}
    assert ("3d_anime_game_general" in ids) or ("3d_honkai_like" in ids) or ("3d_genshin_like" in ids)


def test_suggest_popular_style_presets_respects_category_filter():
    rows = prompt_lexicon.suggest_popular_style_presets(
        "ink lineart comic noir",
        category="drawing",
        limit=6,
    )
    assert rows
    assert all(r["category"] == "drawing" for r in rows)


def test_resolve_artist_technique_controls_pack_and_override():
    out = prompt_lexicon.resolve_artist_technique_controls(
        artist_technique_pack="pbr_3d_master",
        composition_technique="leading_lines",
    )
    assert out["rendering_technique"] == "pbr_3d"
    assert out["material_technique"] == "metal_surface"
    assert out["composition_technique"] == "leading_lines"


def test_artist_technique_bundle_contains_expected_fragments():
    out = prompt_lexicon.artist_technique_bundle(
        linework_technique="crosshatch_precision",
        rendering_technique="toon_3d",
        shading_technique_plan="rim_bounce",
        material_technique="fabric_folds",
        composition_technique="depth_layers",
    )
    assert "precision crosshatch line rhythm" in out
    assert "toon 3d render technique" in out
    assert "rim-plus-bounce lighting technique" in out
    assert "fabric rendering technique" in out
    assert "foreground-midground-background layering technique" in out


# ==================================================================================================
# test_prompt_research: Tests for RAG-based prompt research.
# ==================================================================================================


def test_seed_from_description_uses_first_sentence():
    assert _seed_from_description("A girl in a field. More text.", "") == "A girl in a field"


def test_retrieve_rag_facts_from_corpus(tmp_path):
    corpus = tmp_path / "rag.jsonl"
    rows = [
        {"text": "hatsune miku, vocaloid, 1girl, turquoise hair, stage"},
        {"text": "landscape, mountains, sunset, clouds"},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    facts = retrieve_rag_facts("hatsune miku vocaloid stage", corpus, top_k=2)
    assert facts
    assert any("miku" in f.lower() for f in facts)


def test_research_without_models_uses_fallback(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"not-a-real-png")
    corpus = tmp_path / "rag.jsonl"
    corpus.write_text(
        json.dumps({"text": "1girl, solo, blue hair, outdoors"}) + "\n",
        encoding="utf-8",
    )

    fake = PromptResearchResult(
        diffusion_prompt="1girl, solo, blue hair, outdoors, detailed",
        image_description="A girl with blue hair outdoors.",
        retrieved_facts=["1girl, solo, blue hair, outdoors"],
        sources=["vlm_uncensored", "rag:1", "creative_rag"],
    )

    with patch("utils.captioning.describe_image_uncensored", return_value=fake.image_description):
        with patch("utils.captioning.retrieve_rag_facts", return_value=fake.retrieved_facts):
            with patch("utils.prompt.rag.CreativeRAGEngine", side_effect=RuntimeError("no gpu")):
                result = research_prompt_for_image(
                    img,
                    rag_corpus=corpus,
                    use_creative_rag=True,
                )
    assert "blue hair" in result.diffusion_prompt.lower()
    assert result.retrieved_facts


# ==================================================================================================
# test_prompt_stack: Tests for utils.prompt.stack (PromptStack v2).
# ==================================================================================================


def test_split_and_append_unique():
    assert split_tags("a, b, a") == ["a", "b", "a"]
    assert "c" in split_tags(append_unique("a, b", ["b", "c"]))


def test_analyze_prompt_complexity():
    short = analyze_prompt("1girl, red dress")
    assert short.complexity == "simple"
    long = analyze_prompt(", ".join([f"tag{i}" for i in range(25)]))
    assert long.complexity in ("complex", "extreme")


def test_clause_registry_uncensored():
    pos, neg = apply_clauses("", "", ["uncensored.fidelity"])
    assert "censorship" in neg.lower() or "censor" in neg.lower()
    assert "faithful" in pos.lower()
    assert "uncensored.fidelity" in CLAUSE_REGISTRY
    assert len(list_clauses()) >= 4


def test_style_chaos_clauses():
    pos, neg = apply_clauses("portrait", "low quality", ["style.chaos", "style.glitch"])
    assert "unforgettable" in pos.lower() or "uncanny" in pos.lower()
    assert "AI slop" in neg or "ai slop" in neg.lower() or "template" in neg.lower()


def test_resolve_content_controls_infer_view():
    args = SimpleNamespace(
        pose_mode="none",
        view_angle="none",
        subject_sex="none",
        scene_domain="none",
        clothing_mode="none",
        background_mode="none",
        people_layout="none",
        relationship_mode="none",
        object_layout="none",
        hand_mode="none",
        pose_naturalness="none",
        typography_mode="none",
        quality_pack="none",
        lighting_mode="none",
        skin_detail_mode="none",
        body_proportion="none",
        interaction_intensity="none",
        style_mode="none",
        composition_mode="none",
        artist_composition="none",
        anti_ai_pack="none",
        human_media_mode="none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
        adherence_pack="none",
        style_lock=False,
        anti_style_bleed=False,
        anti_duplicate_subjects=False,
        anti_perspective_drift=False,
        cleanup_conflicting_tags=False,
        text_in_image=False,
        one_shot_boost=False,
        auto_content_fix=True,
        lora=None,
    )
    state = resolve_content_controls(args, "city street, two point perspective, vanishing point")
    assert state.artist_composition == "perspective"


def test_preview_stack_runs_stages():
    args = SimpleNamespace(
        prompt="1girl, portrait",
        negative_prompt="",
        shortcomings_mitigation="none",
        shortcomings_2d=False,
        art_guidance_mode="none",
        anatomy_guidance="none",
        style_guidance_mode="none",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=False,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        realism_autopilot=False,
        quality_pack="top",
        adherence_pack="none",
        auto_content_fix=True,
        prompt_clauses="",
        prompt_stack_intelligence=True,
        prompt_stack_auto_quality=True,
        one_shot_boost=True,
        less_ai=False,
        lora=[],
        text_in_image=False,
        no_neg_filter=False,
        scene_domain="none",
        view_angle="none",
        pose_mode="none",
        style_mode="none",
        clothing_mode="none",
        composition_mode="none",
        artist_composition="none",
        people_layout="none",
        hand_mode="none",
        lighting_mode="none",
        anti_ai_pack="none",
        human_media_mode="none",
        lora_scaffold="none",
        lora_scaffold_auto=False,
    )
    ctx = PromptContext(positive="1girl, portrait", mode=StackMode.PREVIEW, args=args)
    result = run_prompt_stack(ctx)
    assert result.positive
    assert result.negative
    assert "content_controls" in result.trace
    assert any("neg_filter" in step for step in result.trace)


def test_merge_guidance_for_training_caption():
    out = merge_guidance_for_training_caption(
        "concept art environment, matte painting",
        shortcomings_mode="auto",
    )
    assert out.strip()


# ==================================================================================================
# test_prompt_stack_exports: Regression: public prompt stack and config exports.
# ==================================================================================================


def test_stack_exports_append_csv_and_token_set():
    from utils.prompt.stack import append_csv, token_set

    assert "b" in token_set("A, b")
    assert append_csv("a", "b") == "a, b"


def test_config_prompt_domains_canonical():
    from config.prompts import DEFAULT_NEGATIVE_PROMPT

    assert DEFAULT_NEGATIVE_PROMPT


# ==================================================================================================
# test_prompt_train_sample_parity: Train/sample prompt guidance parity and emphasis wiring.
# ==================================================================================================


def test_training_guidance_auto_adds_fragments():
    pos, neg = apply_training_guidance_pair(
        "1girl, solo, anime style",
        "",
        shortcomings_mode="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
    )
    assert len(pos) > len("1girl, solo, anime style")
    assert isinstance(neg, str)


def test_emphasis_strip_and_token_weights():
    raw = "(masterpiece:1.2), 1girl, [bad:0.8]"
    cleaned, segments = parse_prompt_emphasis(raw)
    assert "(" not in cleaned
    assert "[" not in cleaned
    assert "1girl" in cleaned
    assert segments
    import torch

    class _Tok:
        def __call__(self, texts, **kwargs):
            text = texts[0] if isinstance(texts, list) else texts
            words = text.split()
            offsets = []
            pos = 0
            for w in words:
                offsets.append((pos, pos + len(w)))
                pos += len(w) + 2
            while len(offsets) < kwargs.get("max_length", 16):
                offsets.append((0, 0))
            return {
                "input_ids": [[0] * len(offsets)],
                "offset_mapping": [offsets],
            }

    w = token_weights_from_cleaned_segments(
        cleaned, segments, _Tok(), max_length=16, device=torch.device("cpu"), dtype=torch.float32
    )
    assert w is not None
    assert w.numel() >= 1


def test_sample_stack_guidance_matches_training_path():
    """sample.py guidance flags use the same guidance module as training captions."""
    from utils.prompt.stack import apply_sample_prompt_stack

    args = SimpleNamespace(
        prompt="1girl, solo, digital painting",
        negative_prompt="",
        shortcomings_mitigation="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=False,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        human_media_mode="none",
        realism_autopilot=False,
        invent_styles=0,
        style_genome_file="",
        _prompt_layout_negative="",
        _multi_instance_negative="",
        _detailed_scene_negative="",
        _visual_design_negative="",
    )
    train_pos, train_neg = apply_training_guidance_pair(
        args.prompt,
        "",
        shortcomings_mode="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
    )
    sample_pos, sample_neg = apply_sample_prompt_stack(args, args.prompt)
    assert len(sample_pos) >= len(args.prompt)
    assert len(train_pos) >= len(args.prompt)
    assert isinstance(sample_neg, str)
    assert isinstance(train_neg, str)


# ==================================================================================================
# test_prompt_training_pkg_lazy: Importing a ``utils.*`` package (or ``vit_quality``) must stay cheap: no eager submodule imports.
# ==================================================================================================

# OpenMP duplicate-runtime abort (MKL + other libs) in one-shot ``python -c`` smokes.
_SUBPROC_ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], cwd=str(_REPO_ROOT), check=True, env=_SUBPROC_ENV)


def test_utils_packages_import_no_submodules() -> None:
    _run(
        r"""
import importlib
import sys

pkgs = ["utils"] + [
    f"utils.{p}"
    for p in ("agentic", "generation", "modeling", "prompt", "quality", "training")
]
for pkg in pkgs:
    importlib.import_module(pkg)
extra = sorted(k for k in sys.modules if k.startswith("utils.") and k not in pkgs)
assert not extra, f"unexpected eager loads: {extra}"
"""
    )


def test_vit_quality_import_does_not_load_sibling_modules() -> None:
    _run(
        r"""
import sys
import vit_quality as vq

loaded = sorted(k for k in sys.modules if k.startswith("vit_quality.") and k != "vit_quality")
assert loaded == [], f"unexpected eager loads: {loaded}"
_ = vq.ViTConfig
assert "vit_quality.model" in sys.modules
"""
    )


def test_utils_native_does_not_import_sdx_native_until_first_attr() -> None:
    _run(
        r"""
import sys
import utils.native as n

assert "sdx_native" not in sys.modules
_ = n.native_stack_status
assert "sdx_native" in sys.modules
"""
    )


# ==================================================================================================
# test_artist_composition_controls
# ==================================================================================================


def test_artist_composition_lite_adds_tags():
    p, n = apply_content_controls(
        "1girl",
        "bad",
        artist_composition="lite",
    )
    assert "rule of thirds" in p.lower()
    assert "muddy focal point" in n.lower()


def test_infer_artist_composition_from_keywords():
    inf = infer_content_controls_from_prompt("city street, two point perspective, vanishing point")
    assert inf.get("artist_composition") == "perspective"
    inf2 = infer_content_controls_from_prompt("golden ratio portrait, classical composition")
    assert inf2.get("artist_composition") == "classical"


# ==================================================================================================
# test_auto_oc: Tests for utils.prompt.scene.
# ==================================================================================================


def test_prompt_requests_original_character():
    assert prompt_requests_original_character("create an original character for my comic")
    assert prompt_requests_original_character("my oc in a sci-fi city")
    assert not prompt_requests_original_character("landscape with mountains and clouds")


def test_infer_auto_original_character_is_deterministic():
    p = "design an oc space pilot hero"
    a = infer_auto_original_character(p, seed=42)
    b = infer_auto_original_character(p, seed=42)
    assert a is not None and b is not None
    assert a == b


def test_infer_auto_original_character_keyword_bias():
    p = "create an original character detective noir style"
    prof = infer_auto_original_character(p, seed=7)
    assert prof is not None
    assert prof.archetype == "noir_detective"
    assert "original character" in prof.to_prompt_block().lower()


def test_infer_auto_original_character_style_context_bias():
    p = "create an original character for my project"
    prof = infer_auto_original_character(p, seed=5, style_context="anime 3d game toon pbr")
    assert prof is not None
    assert prof.archetype == "space_pilot"
    assert ("stylized topology-friendly hair silhouette" in prof.visual_traits) or (
        "clean anime face planes" in prof.visual_traits
    )


# ==================================================================================================
# test_composition_brief: Tests for utils.prompt.art_direction (layout / text fidelity heuristics).
# ==================================================================================================


def test_auto_skips_plain_prompt():
    assert not composition_brief_warranted("a red apple on a table")
    p = apply_composition_brief("a red apple on a table", "auto")
    assert "coherent perspective" not in p


def test_auto_triggers_poster():
    assert composition_brief_warranted("concert poster with bold title")
    p = apply_composition_brief("concert poster", "auto")
    assert "coherent perspective" in p


def test_auto_triggers_quoted_string():
    assert composition_brief_warranted('sign that says "OPEN" in neon')
    p = apply_composition_brief('sign that says "OPEN"', "auto")
    assert "coherent perspective" in p


def test_on_always_appends():
    p = apply_composition_brief("a red apple", "on")
    assert "coherent perspective" in p


def test_off_noop():
    p = apply_composition_brief("concert poster", "off")
    assert p == "concert poster"


def test_no_double_append_on():
    once = apply_composition_brief("mountain vista", "on")
    twice = apply_composition_brief(once, "on")
    assert twice == once


# ==================================================================================================
# test_creature_character_prompts: Tests for creature / anthro / robot / monster prompt defaults and analyzer.
# ==================================================================================================


def test_suggest_creature_anthro() -> None:
    pos, neg = suggest_creature_prompt_addons("anthropomorphic red fox wearing a jacket")
    assert "anthropomorphic" in pos.lower() or "anthro" in pos.lower()
    assert neg


def test_suggest_creature_robot() -> None:
    pos, neg = suggest_creature_prompt_addons("mecha pilot standing next to building-sized robot")
    assert "robot" in pos.lower() or "hard-surface" in pos.lower() or "panel" in pos.lower()
    assert neg


def test_suggest_creature_empty() -> None:
    pos, neg = suggest_creature_prompt_addons("plain red apple on white table")
    assert pos == ""
    assert neg == ""


def test_suggest_creature_sfw_rating() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro rabbit mage, fantasy illustration", rating="sfw")
    assert "sfw" in pos.lower() or "general audience" in pos.lower()
    assert "nsfw" in neg.lower() or "nude" in neg.lower()


def test_suggest_creature_nsfw_auto() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro wolf, explicit, adult character portrait")
    assert "adult" in pos.lower() or "mature" in pos.lower()
    assert "minor" in neg.lower() or "underage" in neg.lower()


def test_suggest_creature_nsfw_rating() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro cat, detailed", rating="nsfw")
    assert "adult" in pos.lower() or "mature" in pos.lower()
    assert "melted anatomy" in neg.lower() or "fused limbs" in neg.lower()


def test_humanoid_keyword_tuple_covers_succubus() -> None:
    assert "succubus" in HUMANOID_MONSTER_PROMPT_KEYWORDS
    assert "tiefling" in HUMANOID_MONSTER_PROMPT_KEYWORDS


def test_suggest_succubus_triggers_humanoid_pack() -> None:
    pos, neg = suggest_creature_prompt_addons("succubus character with bat wings and tail")
    assert pos and "humanoid" in pos.lower()
    assert neg


# ==================================================================================================
# test_detailed_scene_entities: Tests for detailed multi-entity / pose / physics prompt boosting.
# ==================================================================================================


def test_warrant_group():
    assert detailed_scene_warrants_boost("a crowd of knights in the square")


def test_warrant_count():
    assert detailed_scene_warrants_boost("five people waiting at a bus stop")


def test_no_warrant_short():
    assert not detailed_scene_warrants_boost("a red apple")


def test_apply_on_adds_pose_and_negatives():
    p, neg = apply_detailed_scene_boost(
        "two warriors dueling on a stone bridge over water",
        "on",
        strength="lite",
    )
    assert "water" in p.lower() or "bridge" in p.lower()
    assert "meniscus" in p.lower() or "liquid" in p.lower() or "spatially" in p.lower()
    assert neg
    assert "merged" in neg.lower() or "floating" in neg.lower()


def test_extract_key_segments():
    s = extract_key_segments("masterpiece, a tall knight in armor, glowing sword")
    assert any("knight" in x.lower() for x in s)


def test_off_noop_detailed_scene_entities():
    p, neg = apply_detailed_scene_boost("crowd scene", "off")
    assert p == "crowd scene" and neg == ""


def test_auto_skips_when_not_warranted():
    p, neg = apply_detailed_scene_boost("a red apple on a table", "auto")
    assert p == "a red apple on a table" and neg == ""


# ==================================================================================================
# test_multi_instance_scene: Tests for multi-instance scene prompting (posters, stacks, turnarounds).
# ==================================================================================================


def test_none_is_noop():
    p, neg, n, ec, note = apply_multi_instance_preset("a cat", "none")
    assert p == "a cat" and neg == "" and n == 1 and ec is None


def test_distinct_objects_augment():
    p, neg, n, ec, note = apply_multi_instance_preset(
        "wall of concert posters",
        "distinct_objects",
        user_expected_count=0,
    )
    assert "multiple separate framed" in p.lower()
    assert "cloned identical posters" in neg
    assert n >= 8
    assert ec is None
    assert note


def test_user_count_sets_expected_hint():
    _p, _neg, _n, ec, _note = apply_multi_instance_preset(
        "five ads",
        "distinct_objects",
        user_expected_count=5,
    )
    assert ec == 5


def test_turnaround_suggested_count():
    _p, _neg, _n, ec, _note = apply_multi_instance_preset(
        "hero design",
        "turnaround_sheet",
        user_expected_count=0,
    )
    assert ec == 4
    assert "turnaround model sheet" in _p.lower()


def test_limitation_blurb_nonempty():
    assert "memory" in describe_limitation_short().lower()


def test_panel_strip_augment():
    p, neg, n, _ec, _note = apply_multi_instance_preset("my comic", "panel_strip")
    assert "storyboard strip" in p.lower()
    assert "panels melting" in neg
    assert n >= 8


def test_auto_settings_people_vs_objects():
    assert multi_instance_auto_settings("group_portrait")["expected_count_target"] == "people"
    assert multi_instance_auto_settings("distinct_objects")["expected_count_target"] == "objects"
    assert multi_instance_auto_settings("group_portrait")["pick_best"] == "combo_count"


def test_print_hints_smoke():
    buf = io.StringIO()
    print_multi_instance_hints("stacked_media", stream=buf)
    assert "dissect-refs" in buf.getvalue().lower()


# ==================================================================================================
# test_photo_realism: Tests for utils.prompt.art_direction helpers.
# ==================================================================================================


def test_infer_photo_realism_controls_from_prompt():
    out = infer_photo_realism_controls("cinematic still with teal-orange grade and pro mist filter at golden hour")
    assert out.get("photo_realism_pack") == "cinematic"
    assert out.get("photo_color_grade") == "teal_orange"
    assert out.get("photo_filter") == "pro_mist"
    assert out.get("photo_lighting_technique") == "golden_hour"


def test_photo_realism_fragments_builds_positive_and_negative():
    pos, neg = photo_realism_fragments(
        photo_realism_pack="studio_portrait",
        photo_color_grade="kodak_portra",
        photo_lighting_technique="rembrandt",
        photo_filter="clean_digital",
        photo_grain_style="fine_35mm",
        strength=1.0,
    )
    assert "studio portrait realism" in pos
    assert "kodak portra" in pos
    assert "rembrandt portrait lighting technique" in pos
    assert "fine 35mm-like grain structure" in pos
    assert "plastic skin retouch" in neg


def test_is_photographic_prompt_detects_camera_language():
    assert is_photographic_prompt("cinematic still portrait, 35mm lens, golden hour")
    assert not is_photographic_prompt("anime cel shade line art character sheet")


def test_recommend_photo_post_profile_prefers_realism_metric():
    rec = recommend_photo_post_profile(
        photo_realism_pack="film_analog",
        photo_color_grade="cinestill_800t",
        photo_filter="none",
        photo_grain_style="none",
    )
    assert rec["pick_best_metric"] == "combo_realism"
    assert float(rec["photo_post_strength"]) >= 0.7
    assert rec["photo_grain_style"] == "fine_35mm"


# ==================================================================================================
# test_physics_material_prompts: Tests for physics / fluids / transparency prompt defaults.
# ==================================================================================================


def test_suggest_addons_water() -> None:
    pos, neg = suggest_physics_prompt_addons("a cup of water on a table")
    assert "meniscus" in pos or "liquid" in pos.lower()
    assert neg


def test_suggest_addons_glass() -> None:
    pos, neg = suggest_physics_prompt_addons("frosted glass door in office")
    assert "glass" in pos.lower() or "refraction" in pos.lower()
    assert neg


def test_suggest_addons_empty() -> None:
    pos, neg = suggest_physics_prompt_addons("red square abstract")
    assert pos == ""
    assert neg == ""


# ==================================================================================================
# test_prompt_tags: Tests for ``utils.prompt.text`` comma-tag helpers.
# ==================================================================================================


def test_filter_negative_python_matches_contract():
    pos = "red dress, sky"
    neg = "red, blurry, dress, artifact"
    out = filter_negative_by_positive(pos, neg)
    assert "blurry" in out
    assert "red" not in out.split()
    assert "dress" not in out.split()


def test_filter_negative_unified():
    pos = "1girl, blue eyes"
    neg = "1girl, extra limbs, blue"
    out = filter_negative_by_positive(pos, neg)
    assert "extra" in out or "limbs" in out


def test_append_unique_and_merge():
    assert "c" in append_unique("a, b", ["B", "c"])
    m = merge_csv_unique("x, y", "Y, z")
    assert m == "x, y, z"


# ==================================================================================================
# test_rag_prompt_gen_searcher
# ==================================================================================================


def test_facts_from_gen_searcher_payload_extracts_nested_text():
    payload = {
        "reasoning_summary": "Need accurate skyline and bridge orientation.",
        "search_results": [
            {"title": "City skyline guide", "snippet": "Bridge is east-facing at sunset."},
            {"text": "Landmark clocktower has four visible faces."},
        ],
        "references": {"items": [{"description": "Main plaza has red brick pavement."}]},
    }
    facts = rp.facts_from_gen_searcher_payload(payload, max_entries=10)
    assert facts
    joined = " | ".join(facts).lower()
    assert "bridge is east-facing at sunset" in joined
    assert "red brick pavement" in joined


def test_load_facts_from_gen_searcher_json_jsonl(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    rows = [
        {"search_summary": "A", "search_results": [{"snippet": "B"}]},
        {"final_answer": "C"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    facts = rp.load_facts_from_gen_searcher_json(p, max_entries=10)
    assert any(f.lower() == "a" for f in facts)
    assert any(f.lower() == "b" for f in facts)
    assert any(f.lower() == "c" for f in facts)


def test_merge_facts_into_prompt_with_gen_searcher_facts():
    payload = {"search_results": [{"snippet": "Ancient stone walls around the harbor."}]}
    facts = rp.facts_from_gen_searcher_payload(payload, max_entries=4)
    merged = rp.merge_facts_into_prompt("cinematic harbor at dusk", facts, max_chars=600)
    assert "Context (user-supplied facts" in merged
    assert "Ancient stone walls around the harbor." in merged
    assert merged.endswith("cinematic harbor at dusk")
