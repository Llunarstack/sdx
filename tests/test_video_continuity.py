"""Thumbnail rehearsal + continuity validator tests."""

from __future__ import annotations

import pytest
from pipelines.video.continuity_validators import (
    validate_eyeline,
    validate_light_motivation,
    validate_prop_continuity,
)
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict, validate_scene_graph
from pipelines.video.thumbnail_rehearsal import (
    parse_thumbnail_config,
    plan_thumbnails,
    thumbnail_edit_overrides,
    thumbnail_gate_issues,
)


class _Shot:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_thumbnail_config_from_studio():
    cfg = parse_thumbnail_config(None, studio={"thumbnail_first": True, "thumbnail_size": 64, "thumbnail_gate": "warn"})
    assert cfg.enabled
    assert cfg.size == 64
    assert cfg.gate == "warn"


def test_thumbnail_plan_and_gate():
    shots = [
        _Shot(id="a", prompt="hero runs", thumbnail_approved=False),
        _Shot(id="b", prompt="hero lands", thumbnail_approved=True),
    ]
    cfg = parse_thumbnail_config({"thumbnail": {"enabled": True, "gate": "require_approval"}})
    plan = plan_thumbnails(shots, config=cfg, base_prompt="action")
    assert plan.enabled
    assert len(plan.specs) == 2
    assert not plan.gate_passed
    assert thumbnail_gate_issues(plan)


def test_thumbnail_edit_overrides():
    cfg = parse_thumbnail_config({"thumbnail": {"enabled": True}})
    ov = thumbnail_edit_overrides(cfg)
    assert ov["thumbnail_pass"] is True
    assert ov["quality_retry"] is False


def test_eyeline_180_rule():
    shots = [
        _Shot(id="s1", prompt="detective looks left", gaze="frame_left", characters=["det"], transition="cut"),
        _Shot(id="s2", prompt="detective close-up looks left", gaze="frame_left", characters=["det"], transition="cut"),
    ]
    issues = validate_eyeline(shots)
    assert any(i.code == "eyeline_180_rule" for i in issues)


def test_eyeline_dialogue_pair_ok():
    shots = [
        _Shot(id="s1", gaze="frame_right", characters=["a"], transition="cut"),
        _Shot(id="s2", gaze="frame_left", characters=["b"], transition="cut"),
    ]
    issues = validate_eyeline(shots)
    assert not any(i.code == "eyeline_mismatch" for i in issues)


def test_prop_state_reset():
    shots = [
        _Shot(id="s1", objects=["sword"], props_state={"sword": "broken"}),
        _Shot(id="s2", objects=["sword"], props_state={"sword": "pristine"}),
    ]
    ledger = {"sword": {"initial": "drawn"}}
    issues = validate_prop_continuity(shots, ledger)
    assert any(i.code == "prop_state_reset" for i in issues)


def test_light_unmotivated():
    shots = [_Shot(id="s1", prompt="portrait with flat studio lighting")]
    issues = validate_light_motivation(shots)
    assert any(i.code == "light_unmotivated" for i in issues)


def test_scene_compiles_with_continuity_metadata():
    data = {
        "scene": {"prompt": "two heroes talk", "duration_sec": 6},
        "shots": [
            {
                "id": "a",
                "prompt": "hero looks right",
                "gaze": "frame_right",
                "characters": ["hero"],
                "duration_sec": 3,
                "thumbnail_approved": True,
            },
            {
                "id": "b",
                "prompt": "villain looks left",
                "gaze": "frame_left",
                "characters": ["villain"],
                "duration_sec": 3,
                "thumbnail_approved": True,
            },
        ],
        "characters": {
            "hero": {"description": "brave knight"},
            "villain": {"description": "dark sorcerer"},
        },
        "continuity": {"validators": {"eyeline": True, "props": False}},
        "studio": {"thumbnail_first": True, "thumbnail_gate": "off"},
    }
    compiled = compile_scene_graph(parse_scene_dict(data))
    assert compiled.plan.metadata["thumbnail_plan"]["enabled"]
    assert "continuity" in compiled.plan.metadata


def test_thumbnail_gate_blocks_compile():
    data = {
        "scene": {"prompt": "test", "duration_sec": 4},
        "shots": [{"id": "a", "prompt": "wide", "duration_sec": 4, "thumbnail_approved": False}],
        "studio": {"thumbnail_first": True, "thumbnail_gate": "require_approval"},
    }
    with pytest.raises(ValueError, match="Thumbnail gate"):
        compile_scene_graph(parse_scene_dict(data))


def test_validate_scene_graph_continuity_warn():
    data = {
        "scene": {"prompt": "dialogue"},
        "shots": [
            {"id": "a", "gaze": "frame_left", "characters": ["x"]},
            {"id": "b", "gaze": "frame_left", "characters": ["x"]},
        ],
        "continuity": {"validators": {"strict": False}},
    }
    issues = validate_scene_graph(parse_scene_dict(data))
    assert any("eyeline_180_rule" in i for i in issues)
