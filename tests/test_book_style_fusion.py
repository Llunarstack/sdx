"""Tests for pipelines.book_comic.book_style_fusion."""

from __future__ import annotations

from pipelines.book_comic.book_style_fusion import (
    freeform_style_fusion,
    fusion_fragment_from_preset,
    fusion_from_cli,
    primary_style_from_book_type,
)


def test_primary_style_from_book_type():
    assert primary_style_from_book_type("comic") == "comic_us"
    assert primary_style_from_book_type("novel_cover") == "illustration"


def test_fusion_preset_manga_comic():
    s = fusion_fragment_from_preset("manga_comic")
    assert "manga" in s.lower() or "comic" in s.lower() or "hybrid" in s.lower()


def test_fusion_from_cli_secondary_only():
    s = fusion_from_cli(preset="none", secondary="comic_us", primary_book_style="manga")
    assert s
    assert "primary" in s.lower() or "manga" in s.lower()


def test_freeform_empty_secondary():
    assert freeform_style_fusion("manga", "") == ""
