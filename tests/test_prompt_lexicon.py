"""Tests for pipelines.book_comic.prompt_lexicon."""

from pipelines.book_comic.prompt_lexicon import (
    aspect_dimensions,
    enhance_book_prefix,
    merge_prompt_fragments,
    style_snippet,
    suggest_negative_addon,
)


def test_style_snippet_shonen():
    s = style_snippet("shonen")
    assert "speed lines" in s.lower() or "dynamic" in s.lower()


def test_enhance_book_prefix_adds_style():
    out = enhance_book_prefix("manga panel", lexicon_style="chibi", book_type="manga")
    assert "manga panel" in out
    assert "super deformed" in out.lower() or "chibi" in out.lower()


def test_suggest_negative_merges():
    u = suggest_negative_addon(use_lexicon_negative=True, user_negative="bad anatomy")
    assert "bad anatomy" in u
    assert "fingers" in u.lower() or "text" in u.lower()


def test_suggest_negative_production_tier_adds():
    base = suggest_negative_addon(use_lexicon_negative=True, user_negative="")
    extra = suggest_negative_addon(use_lexicon_negative=True, user_negative="", production_tier=True)
    assert len(extra) > len(base)
    assert "moire" in extra.lower() or "cropped" in extra.lower()


def test_enhance_book_prefix_print_and_cover_hints():
    out = enhance_book_prefix(
        "cover",
        lexicon_style="none",
        book_type="novel_cover",
        include_print_finish=True,
        include_cover_spotlight=True,
    )
    assert "print-ready" in out.lower() or "halftone" in out.lower()
    assert "focal" in out.lower() or "typography" in out.lower()


def test_aspect_webtoon_tall():
    w, h = aspect_dimensions("webtoon_tall")
    assert h > w


def test_merge_prompt_fragments():
    assert merge_prompt_fragments("a", "", "b") == "a, b"
