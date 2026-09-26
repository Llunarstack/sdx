"""Tests for permanence / artifact critic / motion grammar / PermanentVideoDiT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from models.permanent_video_dit import (
    PermanentVideoDiT,
    permanence_consistency_loss,
)
from PIL import Image
from pipelines.video.artifact_critic import score_video_artifacts
from pipelines.video.motion_grammar import apply_motion_grammar, grammar_for_prompt
from pipelines.video.permanence import (
    extract_slots,
    repair_permanence_breaks,
    score_object_permanence,
)
from pipelines.video.process_options import parse_process_options
from pipelines.video.quality import score_segment_quality


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_permanence_detects_disappearing_blob(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = 30
        # Bright blob present in first 3 frames, gone after
        if i < 3:
            img[20:36, 20:36] = 220
        p = tmp_path / f"f{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_object_permanence(frames, sample_every=1, k_slots=8)
    assert report.disappear_events >= 1
    assert report.score < 0.95


def test_permanence_repair_reinjects(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 25
        if i == 0:
            img[16:40, 16:40] = 200
        p = tmp_path / f"r{i}.png"
        _save(p, img)
        frames.append(p)
    rep = repair_permanence_breaks(frames, strength=0.9, k_slots=8)
    assert rep.repaired >= 1
    # After repair, later frames should have some mass near the blob
    last = np.asarray(Image.open(frames[-1]))
    assert last[20:36, 20:36].mean() > 40


def test_artifact_critic_flags_flicker(tmp_path: Path):
    frames = []
    for i in range(8):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + (20 if i % 2 == 0 else 200)
        p = tmp_path / f"a{i}.png"
        _save(p, img)
        frames.append(p)
    art = score_video_artifacts(frames)
    assert art.flicker < 0.85
    assert art.overall < 0.95


def test_motion_grammar_product_is_strict():
    g = grammar_for_prompt("ecommerce product turntable SKU packshot")
    assert g.name == "product"
    assert g.min_permanence >= 0.7
    opts = parse_process_options({})
    out = apply_motion_grammar(opts, g)
    assert out.min_permanence >= 0.7
    assert "turntable" in out.motion_grammar_positive or "SKU" in out.motion_grammar_positive


def test_motion_grammar_anime():
    g = grammar_for_prompt("anime girl running, cel shaded")
    assert g.name == "anime_2d"
    assert g.permanence_repair is False


def test_permanent_video_dit_slots_and_loss():
    torch.manual_seed(0)
    model = PermanentVideoDiT(
        in_channels=4, dim=32, depth=2, num_heads=4, patch_size=2, context_dim=16, num_slots=4, permanence_every=1
    )
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.zeros(1)
    out, slots = model(x, t, context=torch.randn(1, 16), return_slots=True)
    assert out.shape == x.shape
    assert slots.shape[0] == 1 and slots.shape[2] == 4
    loss = permanence_consistency_loss(slots)
    assert loss.ndim == 0
    loss.backward()


def test_quality_includes_permanence(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((32, 32, 3), dtype=np.uint8) + 40
        img[8:20, 8:20] = 180
        p = tmp_path / f"q{i}.png"
        _save(p, img)
        frames.append(p)
    q = score_segment_quality(frames, min_permanence=0.2, min_artifact=0.2)
    assert 0.0 <= q.permanence_score <= 1.0
    assert 0.0 <= q.artifact_score <= 1.0


def test_extract_slots_nonempty():
    img = np.zeros((64, 64, 3), dtype=np.uint8) + 10
    img[10:30, 40:55] = 255
    slots = extract_slots(img, k=6)
    assert len(slots) >= 1
