"""Tests for config.defaults.style_artists extraction helpers."""

from config.defaults.style_artists import (
    DIGITAL_ART_STYLE_TAGS_NSFW,
    DIGITAL_ART_STYLE_TAGS_SFW,
    RENDERED_3D_STYLE_TAGS_NSFW,
    RENDERED_3D_STYLE_TAGS_SFW,
    TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_NSFW,
    TRADITIONAL_DRAWN_PAINTED_STYLE_TAGS_SFW,
    append_style_tag_quality_to_prompts,
    compact_style_summary_for_clip,
    describe_style_tag_enrichment,
    detect_style_tag_buckets,
    extract_style_from_text,
    matched_style_facet_ids,
    matching_style_tags_in_prompt,
    prompt_has_style_artists_signal,
    style_embedding_auxiliary_text,
    style_negative_addon,
    style_positive_addon,
    style_tag_quality_fragments,
)


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


def test_describe_style_tag_enrichment_shape():
    d = describe_style_tag_enrichment("packshot_render, studio lighting")
    assert "buckets" in d and "facets" in d
    assert "archviz_product_studio_3d" in d["facets"]
    assert d["positive_fragment"]


def test_addon_and_append_helpers():
    assert prompt_has_style_artists_signal("vector_art, girl")
    assert not prompt_has_style_artists_signal("plain sunset photo")
    p = style_positive_addon("pixel_art, portrait")
    n = style_negative_addon("pixel_art, portrait")
    assert "pixel" in p.lower()
    assert n.strip()
    cap, neg = append_style_tag_quality_to_prompts("1girl, vector_art", "blurry")
    assert "vector_art" in cap
    assert "2d digital" in cap.lower()
    assert "blurry" in neg.lower()


def test_compact_summary_and_style_embed_aux():
    s = compact_style_summary_for_clip("webtoon, clean_lineart, fantasy")
    assert "buckets:" in s and "facets:" in s
    aux = style_embedding_auxiliary_text("artist:wlop, watercolor_painting_traditional")
    assert "wlop" in aux.lower() or "watercolor" in aux.lower()


def test_apply_style_guidance_appends_tag_hints_when_mode_none():
    from data.caption_utils import apply_style_guidance_to_caption_pair

    cap, neg = apply_style_guidance_to_caption_pair(
        "still life, gouache_painting_traditional",
        "blurry",
        mode="none",
        include_artist_refs=True,
    )
    assert "traditional medium" in cap.lower() or "paper tooth" in cap.lower()
    assert "plastic airbrush" in neg.lower() or "smoothness" in neg.lower()

