"""AI film studio architecture tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pipelines.video.animation_principles import preset_principles, principles_prompt
from pipelines.video.camera_rig import parse_camera_rig, rig_to_prompt
from pipelines.video.character_memory import bible_to_prompt, parse_character_bibles
from pipelines.video.director_mode import expand_prompt_to_storyboard, genre_director_notes
from pipelines.video.generation_router import route_scene
from pipelines.video.motion_library import parse_motion_library, resolve_motion_clip
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict
from pipelines.video.studio_compiler import compile_studio_block
from pipelines.video.style_engines import RenderEngine
from pipelines.video.world_memory import parse_world, world_to_prompt


def test_router_anime():
    r = route_scene("a magical girl transforms in tokyo", style_hint="anime")
    assert r.engine == RenderEngine.ANIME_2D


def test_router_minecraft():
    r = route_scene("steve builds a castle in minecraft")
    assert r.engine == RenderEngine.VOXEL


def test_director_mode_expansion():
    exp = expand_prompt_to_storyboard("A detective enters an abandoned factory", duration_sec=8.0)
    assert len(exp.cuts) >= 2
    assert exp.notes.mood
    genre, _ = genre_director_notes("horror creature in dark hallway")
    assert genre == "horror"


def test_animation_principles_pixar():
    p = preset_principles("pixar")
    frag = principles_prompt(p)
    assert "squash" in frag.lower() or "follow" in frag.lower()


def test_world_and_character_memory():
    w = parse_world({"name": "Emberfall", "weather": "ash fog", "palette": "red black"})
    assert w is not None
    assert "ash" in world_to_prompt(w).lower()
    b = parse_character_bibles({"alex": {"hair": "black", "eyes": "green"}})
    assert "black" in bible_to_prompt(b["alex"])


def test_motion_library():
    lib = parse_motion_library({"walk": {"clip": "/nonexistent.mp4"}})
    assert lib.get("walk") is not None
    assert resolve_motion_clip(lib, ["walk"], fallback="fallback.mp4") == "fallback.mp4"


def test_camera_rig():
    r = parse_camera_rig({"preset": "drone", "lens": "24"})
    assert "drone" in rig_to_prompt(r).lower() or "24" in rig_to_prompt(r)


def test_studio_compiler_auto_director():
    out = compile_studio_block(
        {
            "studio": {"engine": "auto", "director_mode": "auto", "director": "spielberg"},
            "scene": {"prompt": "warrior enters temple"},
        },
        scene_prompt="warrior enters temple",
        duration_sec=6.0,
    )
    assert out.engine
    assert out.storyboard_cuts
    assert out.edit


def test_studio_scene_compiles():
    data = {
        "scene": {"prompt": "anime hero leaps across rooftops", "duration_sec": 5},
        "studio": {"engine": "auto", "director_mode": "auto"},
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert compiled.plan.metadata.get("engine")
    assert len(compiled.plan.shots) >= 2


def test_studio_example_file():
    p = Path("examples/scene_studio.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(parse_scene_dict(__import__("json").loads(p.read_text(encoding="utf-8"))))
    assert compiled.plan.metadata.get("engine")
