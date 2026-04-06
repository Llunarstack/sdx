"""Tests for utils.prompt.auto_oc."""

from utils.prompt.auto_oc import infer_auto_original_character, prompt_requests_original_character


def test_prompt_requests_original_character():
    assert prompt_requests_original_character("create an original character for my comic")
    assert prompt_requests_original_character("my oc in a sci-fi city")
    assert not prompt_requests_original_character("landscape with mountains and clouds")


def test_infer_auto_original_character_is_deterministic():
    p = "design an oc space pilot hero"
    a = infer_auto_original_character(p, seed=42)
    b = infer_auto_original_character(p, seed=42)
    assert a is not None and b is not None
    assert a == b


def test_infer_auto_original_character_keyword_bias():
    p = "create an original character detective noir style"
    prof = infer_auto_original_character(p, seed=7)
    assert prof is not None
    assert prof.archetype == "noir_detective"
    assert "original character" in prof.to_prompt_block().lower()


def test_infer_auto_original_character_style_context_bias():
    p = "create an original character for my project"
    prof = infer_auto_original_character(p, seed=5, style_context="anime 3d game toon pbr")
    assert prof is not None
    assert prof.archetype == "space_pilot"
    assert ("stylized topology-friendly hair silhouette" in prof.visual_traits) or (
        "clean anime face planes" in prof.visual_traits
    )

