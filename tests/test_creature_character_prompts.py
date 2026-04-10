"""Tests for creature / anthro / robot / monster prompt defaults and analyzer."""

from __future__ import annotations

from config.defaults import prompt_domains
from config.defaults.creature_character_prompts import (
    CREATURE_CHARACTER_DOMAIN_NAMES,
    HUMANOID_MONSTER_PROMPT_KEYWORDS,
)
from models.complex_prompt_handler import PromptComplexityAnalyzer
from research.creature_character_guidance import suggest_creature_prompt_addons


def test_merged_creature_domains() -> None:
    for key in CREATURE_CHARACTER_DOMAIN_NAMES:
        assert key in prompt_domains.RECOMMENDED_PROMPTS_BY_DOMAIN
        assert key in prompt_domains.RECOMMENDED_NEGATIVE_BY_DOMAIN


def test_suggest_creature_anthro() -> None:
    pos, neg = suggest_creature_prompt_addons("anthropomorphic red fox wearing a jacket")
    assert "anthropomorphic" in pos.lower() or "anthro" in pos.lower()
    assert neg


def test_suggest_creature_robot() -> None:
    pos, neg = suggest_creature_prompt_addons("mecha pilot standing next to building-sized robot")
    assert "robot" in pos.lower() or "hard-surface" in pos.lower() or "panel" in pos.lower()
    assert neg


def test_suggest_creature_empty() -> None:
    pos, neg = suggest_creature_prompt_addons("plain red apple on white table")
    assert pos == ""
    assert neg == ""


def test_suggest_creature_sfw_rating() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro rabbit mage, fantasy illustration", rating="sfw")
    assert "sfw" in pos.lower() or "general audience" in pos.lower()
    assert "nsfw" in neg.lower() or "nude" in neg.lower()


def test_suggest_creature_nsfw_auto() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro wolf, explicit, adult character portrait")
    assert "adult" in pos.lower() or "mature" in pos.lower()
    assert "minor" in neg.lower() or "underage" in neg.lower()


def test_suggest_creature_nsfw_rating() -> None:
    pos, neg = suggest_creature_prompt_addons("anthro cat, detailed", rating="nsfw")
    assert "adult" in pos.lower() or "mature" in pos.lower()
    assert "melted anatomy" in neg.lower() or "fused limbs" in neg.lower()


def test_complexity_analyzer_creature_mode() -> None:
    an = PromptComplexityAnalyzer()
    p = an.analyze("anthro dragon knight, detailed, masterpiece")
    assert p.is_creature_character_heavy is True
    assert p.dominant_mode == "creature"
    assert "anthro" in p.creature_tags or "dragon" in p.creature_tags


def test_humanoid_keyword_tuple_covers_succubus() -> None:
    assert "succubus" in HUMANOID_MONSTER_PROMPT_KEYWORDS
    assert "tiefling" in HUMANOID_MONSTER_PROMPT_KEYWORDS


def test_suggest_succubus_triggers_humanoid_pack() -> None:
    pos, neg = suggest_creature_prompt_addons("succubus character with bat wings and tail")
    assert pos and "humanoid" in pos.lower()
    assert neg
