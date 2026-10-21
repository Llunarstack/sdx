"""Tests for ``utils.visual_design`` prompt packs, presets and ``sample.py`` argv."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from utils.visual_design import (
    DOMAIN_NEGATIVES,
    DOMAIN_POSITIVES,
    VISUAL_DESIGN_PRESETS,
    apply_visual_design_preset_to_prompt,
    apply_visual_design_stage,
    build_visual_design_prompt_pair,
    design_pack_ids,
    extend_sample_argv_visual_design,
    merge_visual_fragments,
    prompt_suggests_domain,
    visual_design_cli_domain_choices,
)


def test_every_domain_has_all_tiers_and_presets_resolve():
    assert set(DOMAIN_POSITIVES) == set(DOMAIN_NEGATIVES)
    for domain in DOMAIN_POSITIVES:
        for tier in ("lite", "standard", "strong"):
            assert DOMAIN_POSITIVES[domain][tier].strip()
            assert DOMAIN_NEGATIVES[domain][tier].strip()
    for domain, _ in VISUAL_DESIGN_PRESETS.values():
        assert domain in DOMAIN_POSITIVES


def test_design_pack_ids_and_cli_choices():
    ids = design_pack_ids()
    for k in (
        "ui_ux",
        "brand",
        "stem",
        "editorial_layout",
        "presentation_slide",
        "technical_blueprint",
        "fashion_flat",
    ):
        assert k in ids
    ch = visual_design_cli_domain_choices()
    assert ch[:2] == ("none", "auto")
    assert "textbook" in ch


def test_merge_visual_fragments_drops_empty():
    assert merge_visual_fragments("a", "", "  b ") == "a, b"


def test_build_pair_appends_fragment_once_and_returns_negative():
    pos, neg = build_visual_design_prompt_pair("hero shot of a soda can", "packaging", intensity="standard")
    assert pos.startswith("hero shot of a soda can, ")
    assert neg
    again, _ = build_visual_design_prompt_pair(pos, "packaging", intensity="standard")
    assert again == pos


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown"):
        build_visual_design_prompt_pair("x", "not_a_domain")


def test_prompt_suggests_domain():
    assert prompt_suggests_domain("Redesign our app settings screen in Figma style") == "ui_ux"
    assert prompt_suggests_domain("minimal wordmark for a coffee roaster") == "brand"
    assert prompt_suggests_domain("patent drawing exploded isometric view") == "technical_blueprint"
    assert prompt_suggests_domain("random abstract splatter") is None


def test_stage_auto_heuristic():
    logs: list[str] = []
    r = apply_visual_design_stage(
        " keynote slide comparing Q2 revenue ",
        cli_domain="auto",
        intensity="lite",
        use_negative_pack=False,
        emit=logs.append,
    )
    assert r.resolved_domain == "presentation_slide"
    assert "keynote slide" in r.prompt.lower()
    assert r.negative_addon == ""
    assert logs


def test_stage_negative_pack_optional():
    on = apply_visual_design_stage("hero bottle render", cli_domain="packaging", use_negative_pack=True)
    off = apply_visual_design_stage("hero bottle render", cli_domain="packaging", use_negative_pack=False)
    assert on.negative_addon and not off.negative_addon


def test_preset_resolve():
    assert apply_visual_design_preset_to_prompt("logo", "logo_lockup") == ("logo", "brand", "standard")
    assert apply_visual_design_preset_to_prompt("logo", "nope") == ("logo", "none", "standard")


def test_extend_sample_argv_preset_only():
    ns = SimpleNamespace(
        visual_design_preset="saas_ui",
        visual_design_domain="none",
        visual_design_intensity="standard",
        visual_design_negative_pack=True,
    )
    cmd: list[str] = []
    extend_sample_argv_visual_design(cmd, ns)
    assert cmd == ["--visual-design-preset", "saas_ui", "--visual-design-negative-pack"]


def test_extend_sample_argv_domain_path():
    ns = SimpleNamespace(
        visual_design_preset="",
        visual_design_domain="stem",
        visual_design_intensity="strong",
        visual_design_negative_pack=False,
    )
    cmd: list[str] = []
    extend_sample_argv_visual_design(cmd, ns)
    assert cmd == ["--visual-design-domain", "stem", "--visual-design-intensity", "strong"]
