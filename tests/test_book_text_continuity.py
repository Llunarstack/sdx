"""Tests for pipelines.book_comic.book_text_continuity."""

from __future__ import annotations

from pipelines.book_comic.book_text_continuity import lettering_visual_memory_fragment, text_continuity_clause


def test_lettering_visual_memory_fragment():
    s = lettering_visual_memory_fragment(
        {
            "balloon_style": "rounded tails",
            "sfx_style": "brush impact",
            "match_quoted_script": True,
        }
    )
    assert "balloon" in s.lower()
    assert "script" in s.lower()


def test_text_continuity_clause():
    s = text_continuity_clause(
        {
            "strict_script": True,
            "locked_phrases": ["I am the storm", "Chapter 7"],
            "object_labels": [{"id": "blade", "label": "VOID"}],
        }
    )
    assert "storm" in s.lower()
    assert "void" in s.lower()
