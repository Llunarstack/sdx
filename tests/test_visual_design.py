"""Tests for ``utils.visual_design`` prompt helpers."""

from __future__ import annotations

import pytest
from utils.visual_design import (
    apply_visual_design_pack,
    build_visual_design_prompt_pair,
    design_pack_ids,
    merge_visual_fragments,
    prompt_suggests_domain,
    visual_design_cli_domain_choices,
)


def test_design_pack_ids_nonempty():
    ids = design_pack_ids()
    assert "ui_ux" in ids
    assert "brand" in ids
    assert "stem" in ids


def test_visual_design_cli_domain_choices_includes_none_auto():
    ch = visual_design_cli_domain_choices()
    assert ch[0] == "none"
    assert ch[1] == "auto"
    assert "textbook" in ch


def test_merge_visual_fragments_drops_empty():
    assert merge_visual_fragments("a", "", "  b ") == "a, b"


def test_apply_visual_design_pack_appends():
    out = apply_visual_design_pack("mobile banking app home", "ui_ux", intensity="lite")
    assert "mobile banking" in out
    assert "UI" in out or "ui" in out.lower()


def test_build_visual_design_prompt_pair_returns_negative():
    pos, neg = build_visual_design_prompt_pair("hero shot of a soda can", "packaging", intensity="standard")
    assert "soda" in pos
    assert neg


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown"):
        apply_visual_design_pack("x", "not_a_domain")


def test_prompt_suggests_domain():
    assert prompt_suggests_domain("Redesign our app settings screen in Figma style") == "ui_ux"
    assert prompt_suggests_domain("minimal wordmark for a coffee roaster") == "brand"
    assert prompt_suggests_domain("random abstract splatter") is None


def test_visual_design_lazy_attr_origin_covers_all():
    """Each ``__all__`` name must map in ``_ATTR_ORIGIN`` (except the ``presets`` submodule)."""
    import utils.visual_design as vd

    for n in vd.__all__:
        if n == "presets":
            continue
        assert n in vd._ATTR_ORIGIN, f"add _ATTR_ORIGIN[{n!r}] in utils/visual_design/__init__.py"
