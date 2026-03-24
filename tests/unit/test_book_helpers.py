"""Tests for book/comic prompt composition helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_compose_book_page_prompt_orders_fragments() -> None:
    from pipelines.book_comic.book_helpers import compose_book_page_prompt

    s = compose_book_page_prompt(
        user_prompt="hero leaps",
        narration_prefix="noir tone",
        consistency_block="same red scarf",
        panel_hint="two panels",
        rolling_context="prior: chase scene",
    )
    assert "noir" in s and "red scarf" in s and "two panels" in s and "prior" in s and "hero leaps" in s


def test_build_rolling_page_context_truncates() -> None:
    from pipelines.book_comic.book_helpers import build_rolling_page_context

    prev = ["a" * 200, "b" * 200]
    ctx = build_rolling_page_context(prev, num_previous=2, max_chars=50)
    assert ctx.startswith("visual continuity")
    assert len(ctx) <= 50


def test_panel_layout_hint() -> None:
    from pipelines.book_comic.prompt_lexicon import panel_layout_hint

    assert "panel" in panel_layout_hint("single").lower() or "composition" in panel_layout_hint("single").lower()
    assert panel_layout_hint("none") == ""
