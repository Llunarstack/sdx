"""Tests for SFW / NSFW / vague / ambiguous / very-long intent helpers."""

from __future__ import annotations

from utils.prompt.intent_helpers import (
    apply_intent_helpers,
    classify_prompt_intent,
    extract_inline_negatives,
    structure_long_prompt,
)
from utils.superior.prompt_expand import expand_prompt_heuristic


def test_vague_woman_gets_a_decidable_scene():
    intent = classify_prompt_intent("a woman")
    assert intent.is_vague
    pos, neg, _ = apply_intent_helpers("a woman", "")
    assert "one clear subject" in pos.lower()
    assert "blank void" in neg.lower()


def test_fox_in_snow_is_not_vague():
    intent = classify_prompt_intent("a red fox in snow")
    assert intent.is_vague is False
    out = expand_prompt_heuristic("a red fox in snow")
    assert "detail" in out.lower() or "lighting" in out.lower()
    assert out.startswith("a red fox in snow")


def test_sfw_linkedin_covers_clothing():
    pos, neg, intent = apply_intent_helpers("linkedin headshot of a product manager", "")
    assert intent.is_sfw_request
    assert "clothed" in pos.lower() or "attire" in pos.lower()
    assert "nude" in neg.lower()


def test_nsfw_adds_anatomy_not_clothing():
    pos, neg, intent = apply_intent_helpers("nude woman lying on a bed, natural window light", "")
    assert intent.is_nsfw
    assert "gravity-correct" in pos.lower() or "coherent body" in pos.lower()
    assert "fully clothed" not in pos.lower()
    assert "melted anatomy" in neg.lower()
    assert "uncensored" in pos.lower() or "no arbitrary censorship" in pos.lower()
    assert "censor" in neg.lower() or "sanitized" in neg.lower()


def test_explicit_act_tags_count_as_nsfw():
    intent = classify_prompt_intent("1girl, 1boy, handjob, penis, uncensored")
    assert intent.is_nsfw
    pos, neg, _ = apply_intent_helpers("1girl, 1boy, handjob, penis", "fully clothed, sfw only")
    assert "fully clothed" not in neg.lower()
    assert "sfw only" not in neg.lower()
    assert "two distinct people" in pos.lower()
    assert classify_prompt_intent("a fox in Essex snow").is_nsfw is False


def test_cyborg_nsfw_skips_natural_skin():
    pos, _neg, intent = apply_intent_helpers(
        "adult cyborg woman, silicone skin, sex bot, milking machine, penis",
        "",
    )
    assert intent.is_nsfw
    low = pos.lower()
    assert "natural skin texture" not in low
    assert "silicone" in low or "metal" in low


def test_minor_never_gets_nsfw_anatomy():
    pos, _neg, intent = apply_intent_helpers("a child standing in a sunny park", "", mode="nsfw")
    assert intent.looks_like_minor
    assert intent.is_nsfw is False
    assert "genital" not in pos.lower()
    assert "fully clothed" in pos.lower()


def test_ambiguous_or_keeps_both_as_distinct():
    pos, neg, intent = apply_intent_helpers("a cat or a dog sitting in a kitchen", "")
    assert intent.is_ambiguous
    assert intent.or_pairs
    low = pos.lower()
    assert "cat" in low and "dog" in low
    assert "hybrid" in neg.lower() or "chimera" in neg.lower() or "fused" in neg.lower()


def test_long_prompt_moves_quality_prefix_and_extracts_without():
    lead = "masterpiece, best quality, 8k, "
    body = (
        "a weathered lighthouse keeper in a yellow raincoat stands on wet black rocks at dusk, "
        "holding a brass lantern in his left hand, a border collie beside him, storm clouds and "
        "a distant sailboat on the horizon, without watermarks or extra fingers. "
        "shot on 35mm film, cinematic lighting"
    )
    text = lead + body
    intent = classify_prompt_intent(text)
    assert intent.is_very_long
    rebuilt, negs = structure_long_prompt(text)
    assert rebuilt.lower().find("lighthouse") < rebuilt.lower().find("masterpiece")
    assert any("watermark" in n.lower() or "finger" in n.lower() for n in negs)
    pos, neg, _ = apply_intent_helpers(text, "")
    assert "lighthouse" in pos.lower()
    # Do not pad long prompts with generic quality fluff.
    assert pos.lower().count("highly detailed") <= text.lower().count("highly detailed")
    assert "watermark" in neg.lower() or "finger" in neg.lower()


def test_quoted_without_is_kept_in_positive():
    prompt = 'a shop sign that says "without fear" at night'
    cleaned, negs = extract_inline_negatives(prompt)
    assert "without fear" in cleaned.lower()
    assert negs == []


def test_expand_skips_already_long_prompt():
    long = " ".join(["word"] * 40) + " natural lighting"
    assert expand_prompt_heuristic(long) == long


def test_intent_off_is_noop():
    raw = "a woman"
    pos, neg, intent = apply_intent_helpers(raw, "blurry", mode="off")
    assert pos == raw
    assert neg == "blurry"
    assert intent.primary == "none"
