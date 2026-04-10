"""Consistency JSON text_continuity merged into positive block."""

from __future__ import annotations

from pipelines.book_comic.consistency_helpers import positive_block_from_mapping


def test_positive_block_includes_text_continuity():
    spec = {
        "character": "a hero",
        "text_continuity": {
            "chapter_motto": "Never look back",
            "locked_phrases": ["Remember the bridge"],
        },
    }
    out = positive_block_from_mapping(spec)
    assert "Never look back" in out
    assert "bridge" in out.lower()
