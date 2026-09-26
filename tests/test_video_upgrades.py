"""Tests for video keyframe quality ports + neural VideoDiT wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pipelines.video.editor import build_sample_cmd_for_keyframe
from pipelines.video.generation_router import route_scene
from pipelines.video.keyframe_pick import pick_keyframe_with_continuity, pixel_continuity_score
from pipelines.video.neural_sample import latents_to_frames_placeholder, load_video_dit, sample_video_latents
from pipelines.video.process_options import parse_process_options
from pipelines.video.style_engines import engine_by_id, list_engines
from pipelines.video.types import KeyframeEditJob
from pipelines.video.video_helpers import (
    SampleOptions,
    apply_cli_overrides_to_process_options,
    apply_video_quality_preset,
    build_keyframe_sample_extras,
    soft_architecture_lock_args,
)


def test_sample_options_build_quality_extras():
    s = SampleOptions(
        num=3,
        pick_best="combo_vit",
        quality_pack="top",
        human_made="standard",
        architecture_lock="auto",
        dual_stage_layout=True,
        holy_grail=True,
        anti_perspective_drift=True,
        boost_quality=True,
    )
    extras = build_keyframe_sample_extras(s, prompt="a red car")
    joined = " ".join(extras)
    assert "--num" in extras and "3" in extras
    assert "--pick-best" in extras and "combo_vit" in extras
    assert "--quality-pack" in extras
    assert "--human-made" in extras
    assert "--dual-stage-layout" in joined
    assert "--holy-grail" in joined
    assert "--boost-quality" in joined


def test_architecture_lock_soft_on_library_prompt():
    s = SampleOptions(architecture_lock="auto")
    locked = soft_architecture_lock_args("grand classical library with balconies", s)
    assert locked.architecture_lock == "on"
    extras = build_keyframe_sample_extras(locked, prompt="grand classical library with balconies")
    assert "--architecture-lock" in extras


def test_build_sample_cmd_includes_quality_flags():
    job = KeyframeEditJob(
        segment_index=0,
        frame_index=0,
        source_frame_path="in.png",
        output_path="out.png",
        prompt="library hall",
    )
    cmd = build_sample_cmd_for_keyframe(
        job,
        ckpt="ckpt.pt",
        sample_options=SampleOptions(num=2, pick_best="combo_architecture", architecture_lock="on"),
    )
    assert "--architecture-lock" in cmd
    assert "on" in cmd
    assert "--pick-best" in cmd
    assert "combo_architecture" in cmd


def test_video_quality_strong_preset():
    opts = parse_process_options({})
    strong = apply_video_quality_preset(opts, "strong")
    assert strong.motion_beat_keyframes is True
    assert strong.depth_interpolate is True
    assert strong.sample.num >= 3
    assert strong.sample.holy_grail is True
    assert strong.max_retries >= 3


def test_cli_override_does_not_wipe_with_none_pick():
    opts = parse_process_options({"sample": {"pick_best": "combo_vit", "num": 2}})
    out = apply_cli_overrides_to_process_options(opts, keyframe_pick_best="none", keyframe_num=0)
    assert out.sample.pick_best == "combo_vit"


def test_temporal_pick_prefers_continuity(tmp_path: Path):
    prev = np.zeros((32, 32, 3), dtype=np.uint8)
    prev[:] = 40
    close = prev.copy()
    close[:] = 45
    far = prev.copy()
    far[:] = 200
    from PIL import Image

    p_prev = tmp_path / "prev.png"
    p_close = tmp_path / "close.png"
    p_far = tmp_path / "far.png"
    Image.fromarray(prev).save(p_prev)
    Image.fromarray(close).save(p_close)
    Image.fromarray(far).save(p_far)

    assert pixel_continuity_score(close, prev) > pixel_continuity_score(far, prev)
    best, scores = pick_keyframe_with_continuity(
        [p_far, p_close],
        previous_keyframe=p_prev,
        quality_scores=[0.9, 0.5],  # far looks "sharper" but continuity should win with high weight
        continuity_weight=0.8,
    )
    assert best == 1
    assert scores[1] >= scores[0]


def test_list_engines_includes_neural():
    ids = {e.id.value for e in list_engines()}
    assert "neural" in ids
    preset = engine_by_id("neural")
    assert preset is not None
    assert preset.pipeline_notes


def test_route_scene_neural_override():
    d = route_scene("anything", engine_override="neural")
    assert d.engine.value == "neural"
    assert d.edit_overrides.get("neural_engine") is True


def test_video_dit_sequence_context():
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=48, depth=2, num_heads=4, patch_size=2, context_dim=64)
    x = torch.randn(1, 4, 4, 16, 16)
    t = torch.randint(0, 1000, (1,))
    out = model(x, t, context=torch.randn(1, 8, 64))
    assert out.shape == x.shape


def test_neural_sample_smoke(tmp_path: Path):
    model = load_video_dit(None, model_name="VideoDiT-S/2", context_dim=32, device="cpu")
    # Shrink: reload tiny for speed
    from models.video_dit import VideoDiT

    model = VideoDiT(in_channels=4, dim=32, depth=1, num_heads=4, patch_size=2, context_dim=16).eval()
    z = sample_video_latents(
        model,
        frames=4,
        height=8,
        width=8,
        steps=2,
        context=torch.zeros(1, 4, 16),
        device="cpu",
    )
    assert z.shape == (1, 4, 4, 8, 8)
    frames = latents_to_frames_placeholder(z, tmp_path, width=64, height=64)
    assert len(frames) == 4
    assert all(p.is_file() for p in frames)


def test_generate_video_list_engines_flag():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "pipelines" / "video" / "scripts" / "generate_video.py"), "--list-engines"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert r.returncode == 0
    assert "neural" in r.stdout.lower() or "realistic" in r.stdout.lower()
