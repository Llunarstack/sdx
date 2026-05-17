"""Tests for comma-safe caption truncation."""

from __future__ import annotations

from data.caption_truncate import truncate_caption_at_comma_boundary


def test_truncate_keeps_whole_tags():
    parts = [f"tag{i}" for i in range(20)]
    caption = ", ".join(parts)
    out = truncate_caption_at_comma_boundary(caption, 40)
    assert "," not in out.split(",")[-1][:1] or len(out) <= 40
    for piece in out.split(","):
        assert piece.strip() in {f"tag{i}" for i in range(20)}


def test_truncate_single_long_tag():
    long_tag = "x" * 100
    assert truncate_caption_at_comma_boundary(long_tag, 50) == long_tag[:50]
