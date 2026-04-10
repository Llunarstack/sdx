"""Tests for pipelines.book_comic.book_style_authenticity."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.book_comic.book_style_authenticity import (
    peek_visual_memory_book_style,
    resolve_authenticity_bundle,
    resolve_effective_medium,
)


def test_peek_visual_memory_book_style(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"version": 1, "book_style": "webtoon", "entities": {}}), encoding="utf-8")
    assert peek_visual_memory_book_style(p) == "webtoon"


def test_resolve_effective_medium_from_book_type():
    assert resolve_effective_medium(medium="auto", book_type="comic") == "comic_us"
    assert resolve_effective_medium(medium="auto", book_type="novel_cover") == "illustration"


def test_resolve_authenticity_bundle_none():
    b = resolve_authenticity_bundle(level="none", medium="manga", book_type="manga")
    assert b["positive"] == "" and b["negative"] == ""


def test_resolve_authenticity_bundle_manga_strong():
    b = resolve_authenticity_bundle(level="strong", medium="manga", book_type="manga")
    assert "manga" in b["positive"].lower() or "ink" in b["positive"].lower()
    assert b["effective_medium"] == "manga"
    assert "synthetic" in b["negative"].lower() or "ai" in b["negative"].lower()
