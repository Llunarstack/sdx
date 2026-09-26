"""Tests for Seedance/Hailuo/Wan leaders stack helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pipelines.video.camera_path_solver import path_to_motion_deltas, path_to_prompt, solve_camera_path
from pipelines.video.in_gen_scene_cuts import apply_scene_cuts, plan_scene_cuts
from pipelines.video.leaders_compile import compile_leaders_stack
from pipelines.video.lip_sync_driver import score_lip_sync
from pipelines.video.motion_intelligence import compile_motion_intelligence, plan_to_prompt_fragments
from pipelines.video.multimodal_ref_bus import parse_multimodal_refs, ref_budget_ok
from pipelines.video.native_audio_track import energy_envelope, plan_native_audio, synthesize_stereo_bed
from pipelines.video.process_options import parse_process_options


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_multimodal_ref_bus_roles(tmp_path: Path):
    img = tmp_path / "hero.png"
    _save(img, np.zeros((16, 16, 3), dtype=np.uint8) + 100)
    pack = parse_multimodal_refs(
        {
            "identity": [str(img)],
            "style": [{"path": str(img), "tag": "noir"}],
            "audios": [],
        }
    )
    assert pack.counts["image"] >= 2
    ok, _ = ref_budget_ok(pack)
    assert ok
    assert pack.by_role("identity")


def test_native_audio_synth_and_energy(tmp_path: Path):
    plan = plan_native_audio('she says "hello" in the rain', duration_sec=2.0)
    assert any(e.kind == "dialogue" for e in plan.events)
    assert any(e.kind == "ambience" for e in plan.events)
    wav = synthesize_stereo_bed(plan, tmp_path / "bed.wav")
    assert wav.is_file()
    env = energy_envelope(wav, fps=12.0, duration_sec=2.0)
    assert len(env) >= 8
    assert float(env.max()) > 0


def test_lip_sync_score(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 40
        # Vary mouth band edges with i
        img[28:38, 14:34] = 40 + (i % 3) * 60
        p = tmp_path / f"l{i}.png"
        _save(p, img)
        frames.append(p)
    ae = np.linspace(0, 1, 6).astype(np.float32)
    rep = score_lip_sync(frames, audio_energy=ae)
    assert 0.0 <= rep.score <= 1.0


def test_scene_cuts_and_camera_path(tmp_path: Path):
    plan = plan_scene_cuts("wide shot then close-up then cut to door", frame_count=48, fps=24)
    assert len(plan.cuts) >= 1
    frames = []
    for i in range(24):
        p = tmp_path / f"c{i}.png"
        _save(p, np.zeros((32, 32, 3), dtype=np.uint8) + 50)
        frames.append(p)
    apply_scene_cuts(frames, plan)
    path = solve_camera_path("hitchcock dolly zoom on the hero")
    assert path.preset == "hitchcock"
    assert "dolly" in path_to_prompt(path).lower() or "vertigo" in path_to_prompt(path).lower()
    deltas = path_to_motion_deltas(path, frames=12)
    assert len(deltas) == 12


def test_motion_intelligence_gates():
    plan = compile_motion_intelligence("two people run through rain holding bags")
    assert plan.primary_action == "running"
    assert "feet_to_ground" in plan.contacts
    assert plan.gate_boosts.get("min_physics", 0) >= 0.5 or "fluid" in str(plan.secondary)
    pos, neg = plan_to_prompt_fragments(plan)
    assert "running" in pos
    assert neg


def test_leaders_compile_bundle():
    opts = parse_process_options({"leaders_stack": True, "native_audio": True})
    bundle = compile_leaders_stack(
        opts,
        prompt="orbit camera around product, then cut to close-up, she says hello",
        duration_sec=5.0,
        multimodal_refs={"style": []},
    )
    assert bundle.audio_plan is not None
    assert bundle.camera_path is not None
    assert bundle.motion_plan is not None
    assert any("native_audio" in o or "camera_path" in o or "motion_intel" in o for o in bundle.ops)
