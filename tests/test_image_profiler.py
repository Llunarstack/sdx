"""Tests for image profiler fusion logic."""

from __future__ import annotations

from pathlib import Path

from utils.caption.image_profiler import profile_image


def test_booru_metadata_wins(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    row = {
        "caption": "1girl, solo, long hair",
        "character_tags": ["hatsune_miku"],
        "copyright_tags": ["vocaloid"],
        "artist_tags": ["kantoku"],
    }
    prof = profile_image(img, booru_row=row, use_reverse_search=False, use_vlm=False)
    assert "hatsune miku" in prof.caption
    assert "vocaloid" in prof.caption
    assert "kantoku" in prof.caption
    assert prof.is_original_character is False
    assert prof.confidence >= 0.9
    assert prof.scene_summary
    assert "hatsune miku" in prof.scene_summary.lower()


def test_oc_when_no_identity(tmp_path):
    img = tmp_path / "y.png"
    img.write_bytes(b"fake")
    row = {"caption": "1girl, solo, blue hair, original"}
    prof = profile_image(img, booru_row=row, use_reverse_search=False, use_vlm=False)
    assert prof.is_original_character is True
    assert prof.scene_summary


def test_metadata_only_without_image_file():
    row = {
        "caption": "1girl, solo, outdoors",
        "character_tags": ["rem_(re:zero)"],
        "copyright_tags": ["re:zero_kara_hajimeru_isekai_seikatsu"],
    }
    prof = profile_image(Path("_missing.png"), booru_row=row, use_reverse_search=False, use_vlm=False)
    assert "rem" in prof.caption
    assert prof.scene_summary
    assert prof.confidence >= 0.9
