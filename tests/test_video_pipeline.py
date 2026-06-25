"""Tests for pipelines/video retrieve → compose pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pipelines.video.helpers import preview_retrieval_rankings
from pipelines.video.interpolate import blend_frames, interpolate_between_keyframes
from pipelines.video.motion import extract_motion_profile
from pipelines.video.retrieval import rank_clips_for_shot
from pipelines.video.shot_planner import plan_video_from_prompt, split_prompt_into_beats
from pipelines.video.timeline import frame_count_for_duration, retime_frame_indices
from pipelines.video.types import ClipCandidate, RetrievalSource, ShotSpec, VideoMode
from pipelines.video.video_io import save_frame_rgb, write_video_from_frames


def test_split_prompt_beats():
    beats = split_prompt_into_beats("wide city at dusk. then close-up portrait")
    assert len(beats) >= 2


def test_plan_video_shots():
    plan = plan_video_from_prompt("establishing city, then portrait close-up", duration_sec=6.0)
    assert plan.mode == VideoMode.T2V
    assert len(plan.shots) >= 2
    assert abs(sum(s.duration_sec for s in plan.shots) - 6.0) < 0.2


def test_rank_clips_for_shot():
    shot = ShotSpec(index=0, prompt="city establishing wide urban", duration_sec=3.0, shot_type="establishing")
    candidates = [
        ClipCandidate(
            source=RetrievalSource.LOCAL,
            path="/a.mp4",
            title="city establishing wide",
            tags=["urban", "pan"],
            duration_sec=5.0,
        ),
        ClipCandidate(
            source=RetrievalSource.LOCAL,
            path="/b.mp4",
            title="portrait studio",
            tags=["face"],
            duration_sec=4.0,
        ),
    ]
    ranked = rank_clips_for_shot(shot, candidates)
    assert ranked[0].path == "/a.mp4"
    assert ranked[0].score >= ranked[1].score


def test_timeline_helpers():
    assert frame_count_for_duration(1.0, 24.0) == 24
    idx = retime_frame_indices(10, 20)
    assert len(idx) == 20
    assert idx[0] == 0
    assert idx[-1] == 9


def test_blend_frames():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.ones((8, 8, 3), dtype=np.uint8) * 255
    mid = blend_frames(a, b, 0.5)
    assert mid.shape == a.shape
    assert 100 < int(mid[0, 0, 0]) < 156


def test_interpolate_between_keyframes(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    save_frame_rgb(a, np.zeros((32, 32, 3), dtype=np.uint8))
    save_frame_rgb(b, np.full((32, 32, 3), 200, dtype=np.uint8))
    mids = interpolate_between_keyframes(a, b, tmp_path / "mid", count=3, use_flow=False)
    assert len(mids) == 3


def test_motion_profile(tmp_path: Path):
    pytest.importorskip("cv2")
    paths = []
    for i in range(4):
        fp = tmp_path / f"f{i}.png"
        arr = np.zeros((48, 48, 3), dtype=np.uint8)
        arr[:, :, 0] = i * 40
        save_frame_rgb(fp, arr)
        paths.append(fp)
    prof = extract_motion_profile(paths)
    assert prof.frame_count == 4


def test_preview_retrieval_empty():
    rows = preview_retrieval_rankings("sunset dragon", catalog_path="nonexistent.json")
    assert isinstance(rows, list)


def test_dry_run_pipeline(tmp_path: Path):
    from pipelines.video.pipeline import run_t2v_pipeline

    out = tmp_path / "out.mp4"
    res = run_t2v_pipeline(
        "a red ball rolling",
        out,
        work_dir=tmp_path / "work",
        dry_run=True,
        duration_sec=1.0,
        fps=12.0,
        width=256,
        height=144,
    )
    assert Path(res.output_path).is_file()
    assert res.plan.shots
    assert res.metadata.get("dry_run") is True


def test_write_video_from_frames(tmp_path: Path):
    frames = []
    for i in range(6):
        fp = tmp_path / f"f{i}.png"
        save_frame_rgb(fp, np.full((64, 64, 3), i * 30, dtype=np.uint8))
        frames.append(fp)
    out = write_video_from_frames(frames, tmp_path / "tiny.mp4", fps=6.0)
    assert out.is_file()
