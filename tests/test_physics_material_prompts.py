"""Tests for physics / fluids / transparency prompt defaults."""

from __future__ import annotations

from config.defaults import prompt_domains
from config.defaults.physics_material_prompts import PHYSICS_MATERIAL_DOMAIN_NAMES
from research.physics_visual_guidance import suggest_physics_prompt_addons


def test_merged_domains_in_prompt_domains() -> None:
    for key in PHYSICS_MATERIAL_DOMAIN_NAMES:
        assert key in prompt_domains.RECOMMENDED_PROMPTS_BY_DOMAIN
        assert key in prompt_domains.RECOMMENDED_NEGATIVE_BY_DOMAIN


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
