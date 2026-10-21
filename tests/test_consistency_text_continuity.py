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


def test_positive_block_ignores_json_nulls():
    spec = {
        "character": {"label": "Ren", "hair": None},
        "props": [{"description": None, "label": "x"}, {"description": "lantern", "label": None}],
        "vehicle": {"description": None},
        "lettering": {"extra": None},
    }
    out = positive_block_from_mapping(spec)
    assert "None" not in out
    assert "recurring character Ren" in out
    assert "same recurring prop: lantern" in out
