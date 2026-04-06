"""Tests for artist-facing helpers in pipelines.book_comic.prompt_lexicon."""

from pipelines.book_comic import prompt_lexicon


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
        nsfw_pack="extreme",
    )
    assert out["artist_pack"] == "webtoon_scroll"
    assert out["safety_mode"] == "nsfw"
    assert out["nsfw_pack"] == "extreme"  # explicit override wins
    assert out["nsfw_civitai_pack"] == "style"


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


def test_infer_auto_humanize_controls_storyboard_and_nsfw():
    sb = prompt_lexicon.infer_auto_humanize_controls(
        book_type="storyboard",
        lexicon_style="none",
        safety_mode="",
    )
    assert sb["humanize_profile"] == "lite"
    ns = prompt_lexicon.infer_auto_humanize_controls(
        book_type="manga",
        lexicon_style="shonen",
        safety_mode="nsfw",
    )
    assert ns["humanize_profile"] == "balanced"


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


def test_style_snippet_supports_extended_style_keys():
    assert "cyberpunk art direction" in prompt_lexicon.style_snippet("cyberpunk")
    assert "ukiyo-e inspired treatment" in prompt_lexicon.style_snippet("ukiyo_e")
    assert "octane-like render style" in prompt_lexicon.style_snippet("render_octane")


def test_style_snippet_supports_huge_additional_keys():
    assert "baroque-inspired visual language" in prompt_lexicon.style_snippet("baroque")
    assert "shonen battle anime style" in prompt_lexicon.style_snippet("anime_shonen_battle")
    assert "archviz realism style" in prompt_lexicon.style_snippet("archviz_real")


def test_resolve_art_medium_controls_with_new_pack():
    out = prompt_lexicon.resolve_art_medium_controls(
        art_medium_pack="sports_photo_pro",
    )
    assert out["family"] == "realistic_art"
    assert out["variant"] == "sports_photo"
