"""Tests for scripts.tools.book_scene_split."""

from scripts.tools.book_scene_split import normalize_one_line, split_into_page_prompts


def test_split_pages_heading():
    raw = """## Page 1
first scene

## Page 2
second scene
"""
    pages = split_into_page_prompts(raw)
    assert len(pages) == 2
    assert "first" in pages[0]
    assert "second" in pages[1]


def test_normalize_one_line():
    assert normalize_one_line("a\n\nb") == "a b"
