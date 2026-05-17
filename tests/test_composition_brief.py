"""Tests for utils.prompt.composition_brief (layout / text fidelity heuristics)."""

from __future__ import annotations

from utils.prompt.composition_brief import (
    apply_composition_brief,
    composition_brief_warranted,
)


def test_auto_skips_plain_prompt():
    assert not composition_brief_warranted("a red apple on a table")
    p = apply_composition_brief("a red apple on a table", "auto")
    assert "coherent perspective" not in p


def test_auto_triggers_poster():
    assert composition_brief_warranted("concert poster with bold title")
    p = apply_composition_brief("concert poster", "auto")
    assert "coherent perspective" in p


def test_auto_triggers_quoted_string():
    assert composition_brief_warranted('sign that says "OPEN" in neon')
    p = apply_composition_brief('sign that says "OPEN"', "auto")
    assert "coherent perspective" in p


def test_on_always_appends():
    p = apply_composition_brief("a red apple", "on")
    assert "coherent perspective" in p


def test_off_noop():
    p = apply_composition_brief("concert poster", "off")
    assert p == "concert poster"


def test_no_double_append_on():
    once = apply_composition_brief("mountain vista", "on")
    twice = apply_composition_brief(once, "on")
    assert twice == once
