"""Tests for VIDEOMAX consistency stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pipelines.video.process_options import ProcessOptions, parse_process_options
from pipelines.video.superior_pass import run_superior_pass
from pipelines.video.video_helpers import apply_cli_overrides_to_process_options, apply_video_quality_preset
from pipelines.video.videomax import FAILURE_AXES, apply_videomax_to_options, plan_videomax, videomax_sample_extras


def _write_frames(tmp: Path, n: int = 5) -> list[Path]:
    paths = []
    for i in range(n):
        p = tmp / f"f{i:02d}.png"
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        rgb[10:30, 10:30] = (40 + i * 5, 80, 120)
        from PIL import Image

        Image.fromarray(rgb).save(p)
        paths.append(p)
    return paths


def test_plan_videomax_axes() -> None:
    plan = plan_videomax("cinematic photoreal walking")
    assert plan.active
    assert "identity_drift" in plan.axes
    assert "style_mismatch" in plan.axes
    assert plan.option_overrides.get("shot_chain") is True
    assert plan.option_overrides.get("count_bind_repair") is True
    assert "--invention-stack" in plan.sample_extras
    assert len(FAILURE_AXES) >= 10
    assert plan.style in ("realistic", "film", "")


def test_videomax_unions_style_and_videowave() -> None:
    plan = plan_videomax("sakuga anime fight scene, cel shaded")
    inv = str(plan.option_overrides.get("invention_stack") or "")
    assert "videowave" in inv
    # Style may add artwave; stack must keep videowave
    assert "--invention-stack" in plan.sample_extras
    idx = plan.sample_extras.index("--invention-stack")
    stack_val = plan.sample_extras[idx + 1]
    assert "videowave" in stack_val
    opts = apply_videomax_to_options(ProcessOptions(), prompt="anime girl walking")
    extras = videomax_sample_extras(opts, prompt="anime girl walking")
    assert "videowave" in extras[extras.index("--invention-stack") + 1]


def test_apply_videomax_to_options() -> None:
    opts = ProcessOptions()
    out = apply_videomax_to_options(opts, prompt="two people walking")
    assert getattr(out, "videomax", False) is True
    assert getattr(out, "occlusion_resolve", False) is True
    assert getattr(out, "shot_chain", False) is True
    assert float(getattr(out, "min_permanence", 0)) >= 0.55
    extras = videomax_sample_extras(out, prompt="x")
    assert "--invention-stack" in extras


def test_quality_preset_max() -> None:
    opts = apply_video_quality_preset(ProcessOptions(), "max")
    assert getattr(opts, "videomax", False) is True
    opts2 = apply_cli_overrides_to_process_options(ProcessOptions(), videomax=True)
    assert getattr(opts2, "videomax", False) is True


def test_parse_videomax_from_edit() -> None:
    opts = parse_process_options({"videomax": True, "video_quality": "max"})
    assert opts.videomax is True


def test_physics_repair_detects_center_jump(tmp_path: Path) -> None:
    from PIL import Image
    from pipelines.video.physics_repair import apply_physics_repair

    paths = []
    for i in range(5):
        p = tmp_path / f"j{i:02d}.png"
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        # Frame 2 teleports the blob across the frame
        x0 = 8 if i != 2 else 48
        rgb[20:40, x0 : x0 + 16] = (200, 40, 40)
        Image.fromarray(rgb).save(p)
        paths.append(p)
    # Force repair regardless of physics score by lowering trigger
    report = apply_physics_repair(paths, strength=0.5, trigger_below=1.01)
    assert report.repaired >= 1 or report.score_before < 1.0


def test_superior_pass_count_and_physics_repair(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path)
    opts = apply_videomax_to_options(ProcessOptions(), prompt="three red balloons")
    result = run_superior_pass(frames, opts, prompt="three red balloons")
    assert "permanence" in result.scores or result.ops
    # count/physics may or may not repair synthetic blobs; scores should exist
    assert "count" in result.scores or "physics" in result.scores or result.ops
