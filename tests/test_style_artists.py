"""Tests for config.defaults.style_artists extraction helpers."""

from config.defaults.style_artists import extract_style_from_text


def test_extract_style_from_booru_artist_prefix():
    out = extract_style_from_text("masterpiece, highres, artist:reiq, anime_3d, game_cg")
    assert out is not None
    assert "reiq" in out.lower()


def test_extract_style_from_known_3d_tag():
    out = extract_style_from_text("best quality, honkai_star_rail_style, anime_3d")
    assert out is not None
    lo = out.lower()
    assert ("honkai star rail style" in lo) or ("anime 3d" in lo)

