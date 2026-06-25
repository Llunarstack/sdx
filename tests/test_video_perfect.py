"""Tests for perfect-video polish modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pipelines.video.identity_lock import composite_with_mask
from pipelines.video.mask_propagate import rasterize_box_mask
from pipelines.video.motion_transfer import motion_template_sequence, retime_frame_list
from pipelines.video.pose_control import pose_control_args, pose_from_rig_boxes, write_pose_control_image
from pipelines.video.post_grade import apply_grade_to_sequence, grade_frame
from pipelines.video.process_options import parse_process_options
from pipelines.video.segment_retry import RetryPolicy
from pipelines.video.video_io import read_frame_rgb, save_frame_rgb


def test_parse_process_options_defaults():
    opts = parse_process_options({})
    assert opts.motion_transfer is True
    assert opts.identity_lock is True
    assert opts.quality_retry is True
    assert opts.max_retries == 2


def test_parse_process_options_from_scene_edit():
    opts = parse_process_options(
        {
            "motion_transfer": False,
            "post_grade": "teal_orange",
            "pose_control": True,
            "max_retries": 3,
        }
    )
    assert opts.motion_transfer is False
    assert opts.post_grade == "teal_orange"
    assert opts.pose_control is True
    assert opts.max_retries == 3


def test_composite_with_mask():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    anchor = np.full((16, 16, 3), 200, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    out = composite_with_mask(frame, anchor, mask, strength=1.0)
    assert int(out[8, 8, 0]) == 200
    assert int(out[0, 0, 0]) == 0


def test_rasterize_box_mask():
    m = rasterize_box_mask(64, 48, (0.25, 0.25, 0.75, 0.75))
    assert m.shape == (48, 64)
    assert m[24, 32] == 1.0
    assert m[0, 0] == 0.0


def test_retime_frame_list():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) + i for i in range(4)]
    out = retime_frame_list(frames, 8)
    assert len(out) == 8


def test_motion_template_sequence(tmp_path: Path):
    anchor = tmp_path / "anchor.png"
    save_frame_rgb(anchor, np.full((32, 32, 3), 128, dtype=np.uint8))
    src = []
    for i in range(4):
        fp = tmp_path / f"src_{i}.png"
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 50
        save_frame_rgb(fp, arr)
        src.append(fp)
    paths = motion_template_sequence(anchor, src, tmp_path / "out", target_count=6, blend_source=0.0)
    assert len(paths) == 6
    assert all(p.is_file() for p in paths)


def test_pose_control(tmp_path: Path):
    parts = [
        ("head", (0.35, 0.05, 0.65, 0.25)),
        ("torso", (0.30, 0.25, 0.70, 0.55)),
        ("legs", (0.30, 0.55, 0.70, 0.95)),
    ]
    rgb = pose_from_rig_boxes(64, 96, parts)
    assert rgb.shape == (96, 64, 3)
    p = write_pose_control_image(parts, tmp_path / "pose.png", width=64, height=96)
    assert p.is_file()
    args = pose_control_args(p)
    assert "--control-type" in args
    assert "pose" in args


def test_post_grade(tmp_path: Path):
    fp = tmp_path / "f.png"
    save_frame_rgb(fp, np.full((32, 32, 3), 120, dtype=np.uint8))
    graded = grade_frame(read_frame_rgb(fp), "vibrant")
    assert graded.shape == (32, 32, 3)
    apply_grade_to_sequence([fp], "muted")
    assert read_frame_rgb(fp).shape == (32, 32, 3)
    apply_grade_to_sequence([fp], "none")
    assert read_frame_rgb(fp).shape == (32, 32, 3)


def test_dry_run_segment_with_process_options(tmp_path: Path):
    from pipelines.video.pipeline import run_t2v_pipeline

    out = tmp_path / "out.mp4"
    res = run_t2v_pipeline(
        "ball rolling",
        out,
        work_dir=tmp_path / "work",
        dry_run=True,
        duration_sec=0.5,
        fps=8.0,
        width=128,
        height=72,
    )
    assert Path(res.output_path).is_file()


def test_scene_edit_block_compiles(tmp_path: Path):
    from pipelines.video.scene_graph import compile_scene_graph, load_scene_graph

    scene = {
        "version": 2,
        "mode": "t2v",
        "scene": {"prompt": "sunset over hills", "duration_sec": 2, "fps": 12},
        "edit": {
            "identity_lock": False,
            "post_grade": "muted",
            "quality_retry": False,
        },
    }
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(scene), encoding="utf-8")
    compiled = compile_scene_graph(load_scene_graph(p))
    assert compiled.plan.metadata["edit"]["post_grade"] == "muted"
    assert compiled.plan.metadata["edit"]["identity_lock"] is False


def test_retry_policy_defaults():
    pol = RetryPolicy()
    assert pol.max_attempts == 2
    assert pol.strength_decay > 0


def test_depth_aware_blend():
    import numpy as np
    from pipelines.video.depth_interpolate import depth_aware_blend, depth_proxy

    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full((16, 16, 3), 200, dtype=np.uint8)
    d = depth_proxy(a)
    assert d.shape == (16, 16)
    mid = depth_aware_blend(a, b, 0.5)
    assert mid.shape == a.shape


def test_deflicker(tmp_path: Path):
    from pipelines.video.deflicker import apply_deflicker

    paths = []
    for i in range(5):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((16, 16, 3), 80 + i * 30, dtype=np.uint8))
        paths.append(fp)
    apply_deflicker(paths, window=3)
    assert read_frame_rgb(paths[2]).shape == (16, 16, 3)


def test_motion_beats(tmp_path: Path):
    pytest.importorskip("cv2")
    from pipelines.video.motion_beats import detect_motion_beats

    paths = []
    for i in range(8):
        fp = tmp_path / f"f{i}.png"
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, i * 3 : min(32, i * 3 + 6), :] = 255
        save_frame_rgb(fp, arr)
        paths.append(fp)
    beats = detect_motion_beats(paths, min_gap=2)
    assert 0 in beats
    assert len(beats) >= 2


def test_frame_enhance():
    from pipelines.video.frame_enhance import enhance_frame

    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = enhance_frame(rgb, amount=0.2)
    assert out.shape == rgb.shape


def test_transition_fx():
    import numpy as np
    from pipelines.video.transition_fx import apply_transition, transition_overlap_frames_fx
    from pipelines.video.types import TransitionType

    a = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]
    b = [np.full((8, 8, 3), 255, dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    out = apply_transition(a, b, TransitionType.FLASH, overlap=1)
    assert len(out) == 3
    assert transition_overlap_frames_fx(TransitionType.WHIP, 24.0) >= 2


def test_parse_process_options_new_flags():
    opts = parse_process_options({"propagate_masks": False, "deflicker": False, "audio_from_source": True})
    assert opts.propagate_masks is False
    assert opts.audio_from_source is True


def test_semantic_drift():
    from pipelines.video.semantic_drift import frame_drift_score

    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert frame_drift_score(b, a) > frame_drift_score(a, a)


def test_drift_repair(tmp_path: Path):
    from pipelines.video.drift_repair import repair_sequence_drift

    paths = []
    for i in range(4):
        fp = tmp_path / f"f{i}.png"
        val = 50 if i != 2 else 220
        save_frame_rgb(fp, np.full((16, 16, 3), val, dtype=np.uint8))
        paths.append(fp)
    report = repair_sequence_drift(paths, threshold=0.3, blend_strength=0.5)
    assert report.repaired_count >= 0


def test_velocity_ease(tmp_path: Path):
    from pipelines.video.velocity_curve import apply_velocity_ease, ease_indices

    idx = ease_indices(10, 10, ease="smooth")
    assert len(idx) == 10
    paths = []
    for i in range(5):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((8, 8, 3), i * 40, dtype=np.uint8))
        paths.append(fp)
    apply_velocity_ease(paths, ease="smooth")
    assert read_frame_rgb(paths[0]).shape == (8, 8, 3)


def test_region_motion_load_boxes(tmp_path: Path):
    from pipelines.video.region_motion import load_rig_boxes

    rig = tmp_path / "rig.json"
    rig.write_text(
        '{"regions": [{"name": "head", "box": [0.3, 0.1, 0.7, 0.3], "reference_mode": "identity"}]}',
        encoding="utf-8",
    )
    boxes = load_rig_boxes(rig)
    assert boxes[0][0] == "head"
    assert boxes[0][2] is True


def test_parallel_segments():
    from pipelines.video.parallel_segments import SegmentWorkItem, run_segments_parallel

    def worker(item: SegmentWorkItem):
        return [Path(f"x{item.index}")], {"ok": True}

    items = [SegmentWorkItem(0, Path(".")), SegmentWorkItem(1, Path("."))]
    out = run_segments_parallel(items, worker, max_workers=2)
    assert len(out) == 2
    assert out[0].index == 0


def test_scene_preflight(tmp_path: Path):
    from pipelines.video.scene_preflight import run_preflight

    scene = tmp_path / "s.json"
    scene.write_text(
        '{"version": 2, "mode": "t2v", "scene": {"prompt": "test", "duration_sec": 2, "fps": 12}}',
        encoding="utf-8",
    )
    report = run_preflight(scene)
    assert report.ok
