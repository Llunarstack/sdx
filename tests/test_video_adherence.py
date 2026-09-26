"""Tests for prompt/image understanding on T2V / I2V / V2V paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pipelines.video.clip_retrieval import rank_clips_structured, score_clip_for_prompt
from pipelines.video.neural_sample import run_neural_segment
from pipelines.video.prompt_ground_graph import bind_at_mentions, parse_prompt_ground
from pipelines.video.temporal_adherence import score_temporal_adherence
from pipelines.video.types import (
    ClipCandidate,
    MasterTimeline,
    RetrievalSource,
    SegmentAssignment,
    ShotSpec,
    VideoMode,
    VideoPlan,
)
from pipelines.video.video_image_cond import image_prompt_brief, image_to_latent_proxy, prepare_i2v_conditioning
from pipelines.video.video_io import read_frame_rgb
from pipelines.video.video_text_encode import encode_prompt_context, null_context


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_prompt_ground_binds_attributes_and_counts():
    g = parse_prompt_ground("a tall woman in a red dress and a man in a blue suit walking, no glasses")
    names = {e.name for e in g.entities}
    assert "woman" in names or "dress" in names
    assert g.rewritten
    assert "attribute" in g.negative_extra.lower() or "swap" in g.negative_extra.lower()
    assert "walk" in g.actions
    g2 = parse_prompt_ground("three red cars parked on the street")
    assert g2.subject_count >= 1 or "car" in {e.name for e in g2.entities} or g2.counts


def test_bind_at_mentions():
    out = bind_at_mentions("hero walks into frame @Image1", {"Image1": "red-haired woman identity"})
    assert "@Image1" in out or "Image1" in out
    assert "red-haired" in out.lower() or "identity" in out.lower()


def test_encode_prompt_context_distinct():
    a = encode_prompt_context("a woman in a red dress", context_dim=64)
    b = encode_prompt_context("a man in a blue suit", context_dim=64)
    assert a.shape[-1] == 64 and a.ndim == 3
    assert not torch.allclose(a.mean(dim=1), b.mean(dim=1), atol=1e-3)
    z = null_context(64)
    assert z.shape[-1] == 64


def test_image_cond_and_i2v_bundle(tmp_path: Path):
    img = tmp_path / "ref.png"
    rgb = np.zeros((64, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    _save(img, rgb)
    brief = image_prompt_brief(img)
    assert brief.has_file and "preserve" in brief.brief.lower()
    lat = image_to_latent_proxy(img, height=16, width=12)
    assert lat.shape == (1, 4, 16, 12)
    bundle = prepare_i2v_conditioning(img, "woman in red dress", latent_h=16, latent_w=12, context_dim=32)
    assert bundle["context"].shape[-1] == 32
    assert bundle["first_frame_latent"].shape[0] == 1


def test_structured_retrieval_prefers_entity_match():
    clips = [
        ClipCandidate(source=RetrievalSource.LOCAL, path="a.mp4", title="ocean waves", tags=["water", "blue"]),
        ClipCandidate(
            source=RetrievalSource.LOCAL, path="b.mp4", title="red sports car", tags=["car", "red", "vehicle"]
        ),
    ]
    prompt = "three red cars racing"
    scored = rank_clips_structured(clips, prompt, top_k=2)
    assert scored[0][0].path == "b.mp4"
    assert score_clip_for_prompt(clips[1], prompt) > score_clip_for_prompt(clips[0], prompt)


def test_temporal_adherence_color_bind(tmp_path: Path):
    frames = []
    for i in range(4):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[8:24, 8:24] = (200, 30, 30)
        p = tmp_path / f"f{i}.png"
        _save(p, rgb)
        frames.append(p)
    rep = score_temporal_adherence(frames, "a woman in a red dress", sample_every=1)
    assert rep.color_bind_score > 0.2
    assert 0.0 <= rep.score <= 1.0


def test_neural_segment_uses_real_context(tmp_path: Path):
    img = tmp_path / "start.png"
    _save(img, np.full((64, 64, 3), 120, dtype=np.uint8))
    shot = ShotSpec(
        index=0,
        prompt="a woman in a red dress walking",
        duration_sec=1.0,
        shot_type="medium",
        motion_hint="walk",
    )
    assignment = SegmentAssignment(shot=shot, clip=None, start_image=str(img))
    plan = VideoPlan(
        mode=VideoMode.I2V,
        user_prompt=shot.prompt,
        timeline=MasterTimeline(fps=8.0, width=64, height=64, duration_sec=1.0),
        shots=[shot],
    )
    frames, meta = run_neural_segment(
        assignment,
        plan,
        tmp_path / "work",
        target_frames=8,
        width=64,
        height=64,
        steps=2,
        device="cpu",
        permanent=True,
    )
    assert len(frames) == 8
    assert "prompt_ground=1" in meta["ops"]
    assert any("context=yes" in o for o in meta["ops"])
    assert any("i2v=yes" in o for o in meta["ops"])
    f0 = read_frame_rgb(frames[0]).astype(np.float32)
    assert float(np.mean(np.abs(f0 - 120))) < 40.0
