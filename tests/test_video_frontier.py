"""Frontier / novel video subsystem tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from frontier.cinema.video_bridge import bridge_shot_prompt, list_video_frontier_modules
from frontier.narrative.tension_field import build_tension_field
from pipelines.video.anticipation_windup import parse_anticipation_config, plan_anticipation_windups
from pipelines.video.causal_events import apply_causal_ripples, detect_triggers_in_prompt, parse_causal_rules
from pipelines.video.frontier_compiler import compile_frontier_layers, list_frontier_modules
from pipelines.video.kinetic_continuity import parse_kinetic_config, track_kinetic_ledger
from pipelines.video.motif_tracker import audit_motif_haunting, parse_motifs
from pipelines.video.narrative_tension import parse_tension_curve, sample_tension_for_shots
from pipelines.video.scene_graph import compile_scene_graph, parse_scene_dict
from pipelines.video.semantic_gravity import build_gravity_field, parse_semantic_gravity
from pipelines.video.temporal_echo import parse_echo_config, plan_temporal_echoes


class _Shot:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_tension_curve_samples():
    curve = parse_tension_curve([0.2, 0.8, 0.5])
    assert curve is not None
    profiles = sample_tension_for_shots(curve, [_Shot(id="a", duration_sec=3)], total_duration=3)
    assert profiles[0].tension > 0


def test_causal_ripple_explosion():
    rules = parse_causal_rules([])
    triggers = detect_triggers_in_prompt("massive explosion in warehouse", rules)
    assert "explosion" in triggers
    ripples = apply_causal_ripples([_Shot(id="x", prompt="massive explosion in warehouse")], rules)
    assert ripples and ripples[0].injected_effects


def test_anticipation_borrow():
    plan = plan_anticipation_windups(
        [_Shot(id="j", prompt="hero jumps across gap", duration_sec=4)],
        config=parse_anticipation_config({"enabled": True}),
    )
    assert plan.windups and plan.windups[0].borrow_sec > 0


def test_motif_haunting_unresolved():
    motifs = parse_motifs({"red_door": {"description": "red door", "must_appear_by_shot": 0}})
    report = audit_motif_haunting(motifs, [_Shot(id="s0", prompt="empty hallway")])
    assert not report.ok
    assert "red_door" in report.unresolved


def test_kinetic_energy_jump():
    ledger = track_kinetic_ledger(
        [
            _Shot(id="a", prompt="sprints at full speed", kinetic={"energy": 0.9}),
            _Shot(id="b", prompt="stands still calmly", kinetic={"energy": 0.05}),
        ],
        config=parse_kinetic_config({"enabled": True}),
    )
    assert any(i.code == "energy_discontinuity" for i in ledger.issues)


def test_semantic_gravity_dominant():
    class Ent:
        lock = True

    field = build_gravity_field(
        {"hero": Ent()},
        {},
        weights=parse_semantic_gravity({"hero": 0.9}),
        shots=[_Shot(characters=["hero"])],
    )
    assert field.dominant_id == "hero"


def test_temporal_echo_auto():
    shots = [
        _Shot(id="a", prompt="neon alley rain cyberpunk city"),
        _Shot(id="b", prompt="interior dialogue scene"),
        _Shot(id="c", prompt="return to neon alley rain cyberpunk pursuit"),
    ]
    plan = plan_temporal_echoes(shots, parse_echo_config({"enabled": True, "auto_rhyme": True}))
    assert plan.links


def test_frontier_compiler_integration():
    data = {
        "scene": {"prompt": "test", "duration_sec": 6},
        "frontier": {"tension_curve": [0.3, 0.9], "breath_cadence": {"enabled": True}},
        "shots": [{"id": "s1", "prompt": "explosion rocks the pier", "duration_sec": 6}],
    }
    out = compile_frontier_layers(data, [_Shot(id="s1", prompt="explosion rocks the pier", duration_sec=6)])
    assert out.metadata.get("tension_curve")
    assert "s1" in out.enrichments


def test_tension_field_sdx():
    plan = build_tension_field(20).plan(0.9)
    assert len(plan.step_emphasis) == 20
    assert plan.cfg_boost > 1.0


def test_video_bridge():
    out = bridge_shot_prompt("samurai about to draw sword in rain", tension=0.8)
    assert "augmented_prompt" in out
    assert out["moment_phase"] in ("anticipation", "static", "climax", "aftermath")


def test_list_frontier_modules():
    assert len(list_frontier_modules()) >= 20
    assert len(list_video_frontier_modules()) >= 20


def test_frontier_example_compiles():
    p = Path("examples/scene_frontier.example.json")
    if not p.is_file():
        pytest.skip("example missing")
    compiled = compile_scene_graph(parse_scene_dict(json.loads(p.read_text(encoding="utf-8"))))
    frontier = compiled.plan.metadata.get("frontier") or {}
    assert frontier.get("tension_curve")
    assert frontier.get("witness_lens") or frontier.get("chromatic_arc")


def test_screen_direction_flip():
    from pipelines.video.screen_direction import parse_screen_direction_config, track_screen_direction

    shots = [
        _Shot(id="a", screen_direction="left", transition="cut"),
        _Shot(id="b", screen_direction="right", transition="cut"),
    ]
    rep = track_screen_direction(shots, config=parse_screen_direction_config({"enabled": True}))
    assert any(i.code == "screen_direction_flip" for i in rep.issues)


def test_weather_inertia_carries():
    from pipelines.video.weather_inertia import parse_weather_config, track_weather_inertia

    rep = track_weather_inertia(
        [_Shot(id="a", prompt="dialogue"), _Shot(id="b", prompt="talk")],
        config=parse_weather_config({"enabled": True, "initial": "rain"}),
    )
    assert rep.prompt_injections.get("a")


def test_chromatic_arc():
    from pipelines.video.chromatic_arc import parse_chromatic_arc, sample_chromatic_for_shots

    beats = sample_chromatic_for_shots(
        parse_chromatic_arc(["hope", "dread", "rage"]),
        [_Shot(id="s", duration_sec=2)],
        total_duration=2,
    )
    assert beats[0].palette_key


def test_witness_lens():
    from pipelines.video.witness_lens import parse_witness_config, plan_witness_lens

    plans = plan_witness_lens([_Shot(id="c", witness="child")], config=parse_witness_config({"enabled": True}))
    assert plans[0].lens == "child"


def test_stinger_frames():
    from pipelines.video.stinger_frames import parse_stinger_config, plan_stinger_frames

    specs = plan_stinger_frames(
        [_Shot(id="x", prompt="hero punches villain")],
        config=parse_stinger_config({"enabled": True}),
    )
    assert specs[0].frame_count >= 1


def test_chromatic_field_sdx():
    from frontier.narrative.chromatic_field import chromatic_field_for_palette

    plan = chromatic_field_for_palette("dread")
    assert plan.positive
