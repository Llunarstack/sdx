"""Tests for architecture / perspective consistency lock."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from utils.quality.architecture_consistency import (
    apply_architecture_consistency,
    build_story_box_layout,
    infer_architecture_profile,
    prompt_asks_novel_view,
    prompt_needs_architecture_lock,
    score_architecture_consistency,
    score_story_depth_order,
    score_vanishing_symmetry,
    synthetic_story_depth_map,
    write_synthetic_depth_control,
)


def test_detects_library_and_novel_view():
    assert prompt_needs_architecture_lock("grand classical library with balconies")
    assert prompt_asks_novel_view("show me the second floor of this library")
    assert not prompt_needs_architecture_lock("a red apple on a table")


def test_infer_profile_stories_and_focus():
    p = infer_architecture_profile("two-story library hall, balcony, vanishing point")
    assert p.stories >= 2
    assert p.has_balcony
    assert p.one_point
    nv = infer_architecture_profile("from the second floor balcony looking down the library aisle")
    assert nv.novel_view or nv.focus_story >= 1


def test_story_box_layout_has_recessed_upper_shelves():
    layout = build_story_box_layout("grand library with balcony and upper bookshelves")
    names = {r["name"] for r in layout["regions"]}
    assert "left_ground_shelves" in names
    assert "left_upper_shelves" in names
    assert "left_balcony_rail" in names
    ground = next(r for r in layout["regions"] if r["name"] == "left_ground_shelves")
    upper = next(r for r in layout["regions"] if r["name"] == "left_upper_shelves")
    # Upper face is inset (farther) → right edge of left-upper is left of ground's right edge.
    assert upper["box"][2] < ground["box"][2]


def test_synthetic_depth_upper_farther_than_rail():
    depth = synthetic_story_depth_map(128, 128, stories=2)
    # near=white
    order = score_story_depth_order(depth, stories=2)
    assert order > 0.55


def test_score_symmetry_prefers_mirror():
    h = w = 64
    good = np.zeros((h, w, 3), dtype=np.uint8)
    good[:, : w // 2] = 40
    good[:, w // 2 :] = 40
    # Strong vertical edges mirrored
    good[:, w // 4] = 200
    good[:, -w // 4] = 200
    bad = np.zeros((h, w, 3), dtype=np.uint8)
    bad[:, :10] = 255
    assert score_vanishing_symmetry(good) >= score_vanishing_symmetry(bad) - 1e-6
    s = score_architecture_consistency(good, "library hall vanishing point")
    assert 0.0 <= s <= 1.0


def test_apply_architecture_consistency_soft_wires(tmp_path):
    args = SimpleNamespace(
        prompt="grand classical library hall with second floor balcony and bookshelves",
        negative_prompt="",
        architecture_lock="on",
        architecture_depth_control=True,
        architecture_depth_strength=0.65,
        anti_perspective_drift=False,
        scene_domain="none",
        artist_composition="none",
        dual_stage_layout=False,
        box_attn_layout=False,
        box_layout="",
        control_image="",
        control=[],
        control_type="auto",
        image_size=256,
        num=3,
        pick_best="none",
        _box_layout_spec=None,
    )
    meta = apply_architecture_consistency(args)
    assert meta["enabled"] is True
    assert args.anti_perspective_drift is True
    assert args.dual_stage_layout is True
    assert args.box_attn_layout is True
    assert args._box_layout_spec is not None
    assert args.control and "depth" in args.control[0]
    assert args.pick_best == "combo_architecture"
    assert "story_box_layout" in meta["applied"]
    assert "synthetic_depth_control" in meta["applied"]


def test_write_depth_png(tmp_path):
    path = write_synthetic_depth_control(tmp_path / "d.png", height=64, width=64, stories=2)
    assert path.is_file()
    from PIL import Image

    arr = np.asarray(Image.open(path))
    assert arr.shape == (64, 64)
