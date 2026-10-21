"""Tests for physics / fluids / transparency prompt defaults."""

from __future__ import annotations

from utils.prompt.subject_addons import suggest_physics_prompt_addons


def test_suggest_addons_water() -> None:
    pos, neg = suggest_physics_prompt_addons("a cup of water on a table")
    assert "meniscus" in pos or "liquid" in pos.lower()
    assert neg


def test_suggest_addons_glass() -> None:
    pos, neg = suggest_physics_prompt_addons("frosted glass door in office")
    assert "glass" in pos.lower() or "refraction" in pos.lower()
    assert neg


def test_suggest_addons_empty() -> None:
    pos, neg = suggest_physics_prompt_addons("red square abstract")
    assert pos == ""
    assert neg == ""
