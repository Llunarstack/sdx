"""Tests for pipelines.book_comic.book_prompt_intel."""

from __future__ import annotations

from pipelines.book_comic.book_prompt_intel import (
    approximate_token_estimate,
    find_cast_mentions,
    panel_layout_hint,
    strip_duplicate_prompt_phrases,
)


def test_approximate_token_estimate_empty():
    assert approximate_token_estimate("") == 0


def test_find_cast_mentions():
    r = find_cast_mentions("Ren talks to Kai near the bike", ["Ren", "Kai", "Yuki"])
    assert "Ren" in r.found and "Kai" in r.found
    assert "Yuki" in r.missing
    assert "Yuki" in r.soft_reminder_fragment()


def test_strip_duplicate_prompt_phrases():
    s = "a cat, a dog, a cat, masterpiece, masterpiece"
    assert strip_duplicate_prompt_phrases(s) == "a cat, a dog, masterpiece"


def test_panel_layout_hint():
    assert "3 distinct" in panel_layout_hint(panels=3, layout="grid", reading_order="left-to-right")
