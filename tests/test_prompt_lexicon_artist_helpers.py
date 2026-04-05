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
