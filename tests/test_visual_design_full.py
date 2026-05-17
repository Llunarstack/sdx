"""Integration-style tests for ``utils.visual_design`` (+ argv + multimodal-facing helpers)."""

from __future__ import annotations

from types import SimpleNamespace

from utils.visual_design import (
    apply_visual_design_stage,
    assert_visual_design_registry_valid,
    design_pack_ids,
    extend_sample_argv_visual_design,
    merge_negative_addon,
    presets,
    prompt_suggests_domain,
    validate_visual_design_registry,
)


def test_registry_validate_passes():
    issues = validate_visual_design_registry()
    assert issues == [], issues
    assert_visual_design_registry_valid()


def test_design_pack_ids_include_extended_domains():
    ids = design_pack_ids()
    for k in ("editorial_layout", "presentation_slide", "technical_blueprint", "fashion_flat"):
        assert k in ids


def test_apply_visual_design_stage_auto_heuristic():
    logs: list[str] = []
    r = apply_visual_design_stage(
        " keynote slide comparing Q2 revenue ",
        cli_domain="auto",
        intensity="lite",
        use_negative_pack=False,
        emit=lambda m: logs.append(m),
    )
    assert r.resolved_domain == "presentation_slide"
    assert "keynote slide" in r.prompt.lower()


def test_apply_visual_design_negative_pack_optional():
    r = apply_visual_design_stage(
        "hero bottle render",
        cli_domain="packaging",
        intensity="standard",
        use_negative_pack=True,
        emit=None,
    )
    assert r.negative_addon


def test_merge_negative_addon_skips_redundant_addon_already_in_base():
    base = "warped label on bottle, extra glare"
    addon = "warped label on bottle"
    merged = merge_negative_addon(base, addon)
    assert merged == base


def test_preset_resolve_logo_lockup():
    p, dom, tier = presets.apply_visual_design_preset_to_prompt("", "logo_lockup")
    assert dom == "brand"
    assert tier == "standard"


def test_prompt_suggests_technical_blueprint():
    assert prompt_suggests_domain("patent drawing exploded isometric view") == "technical_blueprint"


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
    assert "--visual-design-domain stem" in " ".join(cmd)
    assert "--visual-design-intensity strong" in " ".join(cmd)
