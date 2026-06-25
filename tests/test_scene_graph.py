"""Scene graph compile tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipelines.video.scene_graph import (
    compile_scene_graph,
    load_scene_graph,
    parse_scene_dict,
    validate_scene_graph,
)
from pipelines.video.types import VideoMode


def test_simple_scene_auto_shots():
    g = parse_scene_dict({"scene": {"prompt": "sunset over ocean. then lighthouse close-up", "duration_sec": 4}})
    compiled = compile_scene_graph(g)
    assert len(compiled.plan.shots) >= 2
    assert "sunset" in compiled.plan.shots[0].prompt.lower()


def test_cast_and_effects_merge():
    data = {
        "scene": {"prompt": "forest walk", "duration_sec": 3},
        "characters": {"hero": {"description": "red cloak", "lock": True}},
        "objects": {"lantern": "glowing lantern"},
        "shots": [
            {
                "id": "a",
                "prompt": "wide path",
                "duration_sec": 3,
                "characters": ["hero"],
                "objects": ["lantern"],
                "effects": ["fog"],
            }
        ],
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    shot = compiled.plan.shots[0]
    assert "red cloak" in shot.prompt
    assert "glowing lantern" in shot.prompt
    assert "fog" in shot.prompt.lower() or "haze" in shot.prompt.lower()
    assert "hero" in shot.must_preserve


def test_validate_missing_character():
    g = parse_scene_dict(
        {
            "scene": {"prompt": "test"},
            "shots": [{"prompt": "x", "characters": ["nobody"]}],
        }
    )
    issues = validate_scene_graph(g)
    assert any("nobody" in x for x in issues)


def test_example_scene_file_loads():
    p = Path("examples/scene.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(load_scene_graph(p))
    assert compiled.plan.mode == VideoMode.T2V
    assert len(compiled.plan.shots) == 2
    assert compiled.segment_overrides[0]["effects"] == ["fog", "film_grain"]
