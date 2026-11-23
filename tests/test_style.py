"""Tests for config.styles extraction helpers.

Also holds what used to be: test_style_artists, test_style_genome, test_style_guidance, test_style_native.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from config.styles import (
    DIGITAL_ART_STYLE_TAGS_NSFW,
    DIGITAL_ART_STYLE_TAGS_SFW,
    RENDERED_3D_STYLE_TAGS_NSFW,
    RENDERED_3D_STYLE_TAGS_SFW,
    TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW,
    TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW,
    compact_style_summary_for_clip,
    detect_style_ids,
    detect_style_tag_buckets,
    extract_style_from_text,
    matched_style_facet_ids,
    matching_style_tags_in_prompt,
    style_embedding_auxiliary_text,
    style_guidance_fragments,
    style_tag_quality_fragments,
)
from utils.prompt.style import (
    StyleExplorePlanner,
    StyleGenome,
    StyleInventor,
    compile_genome_pair,
    is_genome_novel_enough,
    pick_best_embedding_index,
    style_native_status,
    text_overlap,
)

# ==================================================================================================
# test_style_artists: Tests for config.styles extraction helpers.
# ==================================================================================================


def test_extract_style_from_booru_artist_prefix():
    out = extract_style_from_text("masterpiece, highres, artist:reiq, anime_3d, game_cg")
    assert out is not None
    assert "reiq" in out.lower()


def test_extract_style_from_known_3d_tag():
    out = extract_style_from_text("best quality, honkai_star_rail_style, anime_3d")
    assert out is not None
    lo = out.lower()
    assert ("honkai star rail style" in lo) or ("anime 3d" in lo)


def test_digital_sfw_buckets_non_empty():
    assert len(DIGITAL_ART_STYLE_TAGS_SFW) >= 20
    out = extract_style_from_text("vector_art, flat design, scenery")
    assert out is not None
    assert "vector" in out.lower()


def test_rendered_3d_buckets_non_empty():
    assert len(RENDERED_3D_STYLE_TAGS_SFW) >= 20
    assert len(RENDERED_3D_STYLE_TAGS_NSFW) >= 5
    out = extract_style_from_text("unreal_engine_5, pbr_textures, character")
    assert out is not None
    assert "unreal" in out.lower()


def test_digital_nsfw_style_tag_match():
    assert "hentai" in DIGITAL_ART_STYLE_TAGS_NSFW
    out = extract_style_from_text("1girl, hentai, eroge_cg style")
    assert out is not None


def test_traditional_drawn_painted_buckets():
    assert len(TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW) >= 80
    assert len(TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW) >= 8
    out = extract_style_from_text("study, gouache_painting_traditional, toned paper")
    assert out is not None
    assert "gouache" in out.lower()


def test_detect_style_tag_buckets():
    assert "digital_sfw" in detect_style_tag_buckets("vector_art, scenery")
    assert "rendered_3d_sfw" in detect_style_tag_buckets("unreal_engine_5, hero shot")
    assert "traditional_sfw" in detect_style_tag_buckets("watercolor_painting_traditional")
    assert "digital_nsfw" in detect_style_tag_buckets("1girl, hentai")


def test_style_tag_quality_fragments_nonempty_for_buckets():
    pos, neg = style_tag_quality_fragments("flat design, vector_art, portrait")
    assert "2d digital" in pos.lower()
    assert "focal hierarchy" in pos.lower() or "foreground" in pos.lower()
    assert neg.strip()
    pos3, neg3 = style_tag_quality_fragments("pbr_textures, blender_cycles")
    assert "renderer-coherent" in pos3.lower() or "lighting" in pos3.lower()
    assert "specular" in pos3.lower() or "roughness" in pos3.lower()
    assert neg3.strip()


def test_pixel_facet_adds_discipline_hints():
    pos, neg = style_tag_quality_fragments("pixel_art, 16bit, character portrait")
    assert "pixel" in pos.lower()
    assert "bilinear" in neg.lower() or "softness" in neg.lower()


def test_matching_style_tags_and_facets():
    tags = matching_style_tags_in_prompt("vector_art, webtoon, scenery")
    assert "vector_art" in tags
    assert "webtoon" in tags
    facets = matched_style_facet_ids("webtoon, clean_lineart")
    assert "comic_ink_tone" in facets


def test_compact_summary_and_style_embed_aux():
    s = compact_style_summary_for_clip("webtoon, clean_lineart, fantasy")
    assert "buckets:" in s and "facets:" in s
    aux = style_embedding_auxiliary_text("artist:wlop, watercolor_painting_traditional")
    assert "wlop" in aux.lower() or "watercolor" in aux.lower()


def test_apply_style_guidance_appends_tag_hints_when_mode_none():
    from data.caption_utils import apply_training_guidance_to_caption_pair

    cap, neg = apply_training_guidance_to_caption_pair(
        "still life, gouache_painting_traditional", "blurry", style_guidance_mode="none", style_guidance_artists=True
    )
    assert "traditional medium" in cap.lower() or "paper tooth" in cap.lower()
    assert "plastic airbrush" in neg.lower() or "smoothness" in neg.lower()


# ==================================================================================================
# test_style_genome: Tests for StyleGenome invention and explore planner.
# ==================================================================================================


def test_genome_compile_and_json_roundtrip():
    g = StyleGenome(
        id="test1",
        name="Ash ember wash",
        palette="desaturated umber with ember rim",
        line="dry ink breaks",
        surface="matte gouache on toothy paper",
        camera="low hero angle",
        lighting="single hard key",
        signature="volcanic stillness",
        anti_clone=("not ghibli",),
        negative_fragments=("generic stock photo",),
    )
    pos, neg, style = compile_genome_pair(g, "1girl, forest", "low quality")
    assert "forest" in pos
    assert "ember" in pos.lower() or "gouache" in pos.lower()
    assert "ghibli" in neg.lower()
    assert style

    raw = json.dumps(g.to_dict())
    g2 = StyleGenome.from_dict(json.loads(raw))
    assert g2.id == g.id
    assert g2.palette == g.palette


def test_fallback_inventor_produces_novel_genomes():
    inv = StyleInventor(use_qwen=False)
    genomes = inv.invent("samurai at sunset, dramatic", n=3, seed=99, creativity_level=0.8)
    assert len(genomes) == 3
    names = {g.signature for g in genomes}
    assert len(names) >= 2
    assert all(is_genome_novel_enough(g) for g in genomes)


def test_explore_planner_manifest(tmp_path: Path):
    planner = StyleExplorePlanner(use_qwen=False)
    plan = planner.plan(
        "portrait of a scientist in a lab",
        n_genomes=2,
        mutations_per_genome=1,
        seed=1,
    )
    assert len(plan.candidates) >= 2
    manifest = tmp_path / "manifest.jsonl"
    plan.write_manifest(manifest)
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(plan.candidates)
    row = json.loads(lines[0])
    assert "caption" in row and "style" in row


def test_chaos_fusion_hypermutate():
    from utils.prompt.style import (
        apply_chaos_level,
        fuse_genomes,
        hypermutate,
        invent_insane_batch,
        preset_genome,
    )

    g0 = preset_genome("glitch_cathedral")
    assert g0 is not None
    wild = apply_chaos_level(g0, 0.9)
    assert len(wild.positive_fragments) >= len(g0.positive_fragments)

    batch = invent_insane_batch("cosmic horror library", n=3, mode="insane", chaos_level=0.8)
    assert len(batch) == 3

    chimeras = invent_insane_batch("portrait", n=2, mode="chimera", seed=1)
    assert len(chimeras) == 2
    assert " x " in chimeras[0].name or "Chimera" in chimeras[0].signature

    fused = fuse_genomes(batch[0], batch[1])
    sib = hypermutate(fused, intensity=0.9, seed=99)
    assert sib.id != fused.id
    assert sib.palette


def test_insane_explore_plan():
    from utils.prompt.style import StyleExplorePlanner

    planner = StyleExplorePlanner(use_qwen=False)
    plan = planner.plan(
        "witch in storm",
        n_genomes=2,
        invention_mode="apocalypse",
        chaos_level=0.95,
        fusion_pairs=True,
        hypermutate_siblings=1,
        seed=7,
    )
    kinds = {c.candidate_kind for c in plan.candidates}
    assert "base" in kinds
    assert len(plan.candidates) >= 4


# ==================================================================================================
# test_style_guidance: Tests for config.styles.
# ==================================================================================================


def test_detect_game_style():
    ids = detect_style_ids("fortnite style stylized game art character")
    assert "game_stylized_3d" in ids


def test_detect_anime_style():
    ids = detect_style_ids("anime manga shonen portrait")
    assert "anime_manga" in ids


def test_style_guidance_with_artist_refs():
    pos, neg = style_guidance_fragments(
        "portrait in the style of hayao miyazaki",
        "auto",
        include_artist_refs=True,
    )
    assert "style-faithful" in pos.lower() or "coherent brush" in pos.lower()
    assert "style token drift" in neg.lower()


def test_style_guidance_all_non_empty():
    pos, neg = style_guidance_fragments("test", "all", include_artist_refs=False)
    assert len(pos) > 80
    assert len(neg) > 80


def test_detect_new_style_domains():
    ids = detect_style_ids("baroque surrealist poster graphic with octane render")
    assert "fine_art_movements" in ids
    assert "mixed_media_print_styles" in ids
    assert "render_pipeline_styles" in ids


def test_detect_style_ids_with_aliases():
    ids = detect_style_ids("battle shonen vertical webtoon with bw film look")
    assert "anime_manga" in ids or "anime_specializations" in ids
    assert "comic_webtoon_substyles" in ids
    assert "film_photo_language" in ids


def test_detect_new_color_grade_and_hybrid_styles():
    ids = detect_style_ids("teal orange grade with bleach-bypass mood and anime 3d hybrid framing")
    assert "cinema_color_grading" in ids
    assert "anime_hybrid_rendering" in ids


def test_style_guidance_with_game_and_cartoon_artist_refs():
    pos, neg = style_guidance_fragments(
        "hero shot in riot splash art and cartoon network style",
        "none",
        include_artist_refs=True,
    )
    assert "style-faithful" in pos.lower()
    assert "style token drift" in neg.lower()


def test_detect_anime_game_3d_and_booru_styles():
    ids = detect_style_ids("genshin style anime 3d game render with danbooru tags and masterpiece, highres")
    assert "anime_game_3d_styles" in ids
    assert "booru_2d_3d_prompt_style" in ids


def test_style_guidance_none_includes_tag_bucket_hints():
    pos, neg = style_guidance_fragments(
        "portrait, vector_art, flat design",
        "none",
        include_artist_refs=False,
    )
    assert "2d digital" in pos.lower() or "focal hierarchy" in pos.lower()
    assert neg.strip()


# ==================================================================================================
# test_style_native: Tests for style genome native fast paths (Python fallbacks always run).
# ==================================================================================================


def test_style_native_status_keys():
    st = style_native_status()
    assert "rust_style_ops" in st
    assert "cuda_style_pick" in st
    assert "go_explore" in st


def test_text_overlap():
    o = text_overlap("red dress, blue sky, cinematic", "red dress, golden hour")
    assert 0.0 < o < 1.0


def test_explore_stats_python_fallback(tmp_path):
    import json

    from utils.prompt.style import explore_stats_python

    p = tmp_path / "m.jsonl"
    rows = [
        {"caption": "a", "style_genome_id": "g1", "candidate_kind": "base"},
        {"caption": "b", "style_genome_id": "g1", "candidate_kind": "chimera"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = explore_stats_python(p)
    assert "rows: 2" in out
    assert "g1: 2" in out
    assert "chimera: 1" in out


def test_rust_style_ops_when_built():
    import pytest

    from sdx_native import ffi

    if not ffi.rust_available():
        pytest.skip("sdx_rs release DLL not built (cargo build --release in native/rust)")
    j = ffi.maybe_token_jaccard("a, b", "a, c")
    assert j is not None and j > 0


def test_pick_best_embedding_numpy_fallback():
    dim = 8
    query = np.random.randn(dim).astype(np.float32)
    candidates = np.random.randn(5, dim).astype(np.float32)
    candidates[2] = query + 0.01
    idx, score = pick_best_embedding_index(query, candidates)
    assert 0 <= idx < 5
    assert score > 0.5
