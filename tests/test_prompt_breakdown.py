"""Tests for heuristic prompt breakdown / section reordering."""

from __future__ import annotations

from utils.prompt.prompt_breakdown import (
    apply_prompt_breakdown,
    build_breakdown,
    warrant_prompt_breakdown,
)


def test_warrant_long():
    assert warrant_prompt_breakdown("x" * 120)


def test_warrant_many_clauses():
    assert warrant_prompt_breakdown("a, b, c, d, e")


def test_warrant_short_no():
    assert not warrant_prompt_breakdown("a red apple")
    assert not warrant_prompt_breakdown("")


def test_subject_first_moves_quality_after_subject():
    raw = "masterpiece, 8k, a knight in armor, golden hour"
    res = build_breakdown(raw, order="subject_first")
    assert res.ordered_flat.startswith("a knight in armor")
    assert "masterpiece" in res.ordered_flat
    assert "golden hour" in res.ordered_flat


def test_quality_first_leads_with_quality_tags():
    raw = "a knight in armor, masterpiece, 8k"
    res = build_breakdown(raw, order="quality_first")
    assert res.ordered_flat.startswith("masterpiece")


def test_labeled_format_has_section_labels():
    raw = "a fox in a forest, cinematic lighting, anime style"
    flat, labeled = apply_prompt_breakdown(raw, order="subject_first", output_format="labeled")
    assert flat
    assert "SUBJECTS:" in labeled or "ENVIRONMENT:" in labeled
    assert labeled.count("\n") >= 1


def test_ordered_format_identical_strings():
    raw = "wide angle, a castle, oil painting"
    a, b = apply_prompt_breakdown(raw, output_format="ordered")
    assert a == b
