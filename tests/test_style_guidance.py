"""Tests for config.defaults.style_guidance."""

from config.defaults.style_guidance import detect_style_ids, style_guidance_fragments


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

