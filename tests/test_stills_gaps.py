"""Still-image competitor-gap wiring (auto layout, contact post)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from utils.generation.stills_gaps import apply_auto_layout, apply_contact_post, apply_prompt_ground


def test_auto_layout_left_of_sets_spec() -> None:
    args = SimpleNamespace(
        prompt="A red cube to the left of a blue sphere",
        auto_layout=True,
        no_auto_layout=False,
        glyph_canvas=False,
        no_glyph_canvas=False,
        _box_layout_spec=None,
    )
    apply_auto_layout(args, image_size=256)
    spec = args._box_layout_spec
    assert spec is not None
    assert len(spec.regions) >= 2
    assert spec.regions[0].x2 <= 0.5
    assert spec.regions[1].x1 >= 0.5


def test_auto_layout_merges_occlusion_negative() -> None:
    args = SimpleNamespace(
        prompt="a person behind a chain-link fence",
        auto_layout=True,
        no_auto_layout=False,
        glyph_canvas=False,
        no_glyph_canvas=False,
        negative_prompt="",
        _box_layout_spec=None,
    )
    apply_auto_layout(args, image_size=256)
    assert args._box_layout_spec is not None
    assert "fused" in str(args.negative_prompt).lower() or "occlud" in str(args.negative_prompt).lower()


def test_auto_layout_respects_existing_spec() -> None:
    sentinel = object()
    args = SimpleNamespace(
        prompt="A red cube to the left of a blue sphere",
        auto_layout=True,
        no_auto_layout=False,
        glyph_canvas=False,
        no_glyph_canvas=False,
        _box_layout_spec=sentinel,
    )
    apply_auto_layout(args, image_size=256)
    assert args._box_layout_spec is sentinel


def test_auto_layout_opt_out() -> None:
    args = SimpleNamespace(
        prompt="Exactly three red apples on a plate",
        auto_layout=True,
        no_auto_layout=True,
        glyph_canvas=False,
        no_glyph_canvas=False,
        _box_layout_spec=None,
    )
    apply_auto_layout(args, image_size=256)
    assert args._box_layout_spec is None


def test_prompt_ground_binds_two_subjects() -> None:
    args = SimpleNamespace(
        prompt="a woman in a red dress and a man in a blue suit",
        negative_prompt="",
        prompt_ground=True,
        no_prompt_ground=False,
    )
    apply_prompt_ground(args)
    assert "red" in args.prompt.lower()
    assert "blue" in args.prompt.lower()
    assert "attribute swap" in str(args.negative_prompt).lower() or "woman" in args.prompt.lower()


def test_contact_post_auto_darkens_standing_prompt() -> None:
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    img[40:50, 20:44] = 40
    args = SimpleNamespace(contact_shadow=-1.0, contact_shadow_auto=True, no_contact_shadow=False)
    out = apply_contact_post(img, args, "a woman standing on wet pavement")
    assert out.shape == img.shape
    assert int(out[50:, :, :].mean()) <= int(img[50:, :, :].mean())
