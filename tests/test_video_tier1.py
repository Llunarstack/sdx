"""Tests for Tier-1 video features (elements, storyboard, FLF2V, motion brush)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pipelines.video.elements import parse_elements
from pipelines.video.flf2v import interpolate_flf2v_sequence, prepare_flf2v_keyframes
from pipelines.video.motion_brush import apply_brush_to_flow, parse_motion_brush
from pipelines.video.reference_sheet import build_reference_sheet
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict
from pipelines.video.storyboard import camera_prompt_fragment, parse_storyboard
from pipelines.video.video_io import save_frame_rgb


def test_parse_elements():
    lib = parse_elements(
        {
            "hero": {
                "images": ["a.png", "b.png"],
                "bind_subject": True,
                "reference_sheet": False,
            }
        }
    )
    assert "hero" in lib.elements
    assert lib.elements["hero"].bind_subject is True
    assert len(lib.elements["hero"].images) == 2


def test_reference_sheet(tmp_path: Path):
    src = tmp_path / "hero.png"
    save_frame_rgb(src, np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
    sheet = build_reference_sheet(src, tmp_path / "sheet")
    assert len(sheet.views) == 3
    assert Path(sheet.manifest_path).is_file()


def test_storyboard_parse():
    cuts = parse_storyboard(
        {
            "cuts": [
                {"prompt": "wide city", "duration_sec": 2, "camera": "tracking"},
                {
                    "prompt": "close face",
                    "camera": "push_in",
                    "flf2v": True,
                    "start_image": "a.png",
                    "end_image": "b.png",
                },
            ]
        }
    )
    assert len(cuts) == 2
    assert cuts[1].flf2v is True
    assert "tracking" in camera_prompt_fragment("tracking")


def test_storyboard_compiles_to_shots():
    data = {
        "scene": {"prompt": "fallback", "duration_sec": 6},
        "storyboard": {
            "cuts": [
                {"prompt": "wide", "duration_sec": 3, "camera": "pan_left"},
                {"prompt": "close", "duration_sec": 3, "camera": "close_up"},
            ]
        },
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert len(compiled.plan.shots) == 2
    assert "pan" in compiled.plan.shots[0].prompt.lower() or "pan" in compiled.plan.shots[0].motion_hint.lower()


def test_motion_brush_flow_mask():
    flow = np.ones((8, 8, 2), dtype=np.float32) * 3.0
    brush = np.zeros((8, 8), dtype=np.float32)
    brush[2:6, 2:6] = 1.0
    masked = apply_brush_to_flow(flow, brush, mode="motion_only")
    assert float(masked[0, 0, 0]) == 0.0
    assert float(masked[4, 4, 0]) > 0.0


def test_parse_motion_brush_box():
    spec = parse_motion_brush({"box": [0.1, 0.2, 0.9, 0.8], "mode": "background_only"})
    assert spec is not None
    assert spec.mode == "background_only"


def test_flf2v_interpolate(tmp_path: Path):
    start = tmp_path / "s.png"
    end = tmp_path / "e.png"
    save_frame_rgb(start, np.zeros((32, 32, 3), dtype=np.uint8))
    save_frame_rgb(end, np.full((32, 32, 3), 200, dtype=np.uint8))
    keys = prepare_flf2v_keyframes(start, end, tmp_path / "keys")
    assert len(keys) == 2
    paths = interpolate_flf2v_sequence(start, end, 8, tmp_path / "out", use_flow=False)
    assert len(paths) == 8
    assert paths[0].is_file()


def test_scene_elements_bind(tmp_path: Path):
    hero = tmp_path / "hero.png"
    save_frame_rgb(hero, np.full((48, 48, 3), 128, dtype=np.uint8))
    data = {
        "scene": {"prompt": "walk", "duration_sec": 2},
        "elements": {"hero_el": {"images": [str(hero)], "bind_subject": True}},
        "characters": {"hero": {"bind_element": "hero_el", "description": "knight"}},
        "shots": [{"prompt": "medium shot", "duration_sec": 2, "characters": ["hero"], "bind_elements": ["hero_el"]}],
    }
    data["_work_dir"] = str(tmp_path)
    g = parse_scene_dict(data)
    g.raw["_work_dir"] = str(tmp_path)
    compiled = compile_scene_graph(g)
    assert compiled.control_plans
    assert compiled.control_plans[0].metadata.get("bind_subject") or compiled.control_plans[0].init_image


def test_tier1_example_validates_structure():
    p = Path("examples/scene_tier1.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    g = parse_scene_dict(json.loads(p.read_text(encoding="utf-8")))
    assert g.storyboard
    assert "hero" in g.elements
