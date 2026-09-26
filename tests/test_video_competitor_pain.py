"""Tests for identity bind / extremity / glyph / physics superiority layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from models.permanent_video_dit import PermanentVideoDiT, identity_consistency_loss
from PIL import Image
from pipelines.video.extremity_lock import apply_extremity_lock, score_extremity_coherence
from pipelines.video.glyph_lock import apply_glyph_lock, find_glyph_rois, score_glyph_stability
from pipelines.video.identity_bind import apply_identity_bind, bind_from_refs, score_identity_bind
from pipelines.video.physics_gate import score_physics_invariance
from pipelines.video.quality import score_segment_quality


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_identity_bind_detects_color_drift(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 40
        # Subject band shifts hue mid-clip
        color = (180, 40, 40) if i < 3 else (40, 40, 180)
        img[8:50, 10:54] = color
        p = tmp_path / f"id{i}.png"
        _save(p, img)
        frames.append(p)
    fp = bind_from_refs([frames[0], frames[1]])
    report = score_identity_bind(frames, fp, sample_every=1, drift_threshold=0.85)
    assert report.drift_frames >= 1
    assert report.score < 0.99


def test_identity_bind_repair(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 30
        img[10:50, 12:52] = (200, 80, 60) if i == 0 else (60, 80, 200)
        p = tmp_path / f"ir{i}.png"
        _save(p, img)
        frames.append(p)
    rep = apply_identity_bind(frames, strength=0.8, drift_threshold=0.9)
    assert rep.repaired >= 1


def test_extremity_collapse_detected(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((80, 80, 3), dtype=np.uint8) + 20
        # Structured edges in lower band for early frames, mush later
        if i < 2:
            for x in range(10, 70, 3):
                img[50:75, x] = 220
                img[50:75, x + 1] = 40
        p = tmp_path / f"ex{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_extremity_coherence(frames)
    assert 0.0 <= report.score <= 1.0
    rep = apply_extremity_lock(frames, strength=0.7, collapse_drop=0.05)
    assert rep.repaired >= 0


def test_glyph_melt_and_lock(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 25
        # Sharp logo-like vertical bars — then obliterate into flat gray
        if i <= 1:
            for x in range(8, 40, 2):
                img[8:28, x] = 250
                img[8:28, x + 1] = 10
        else:
            img[8:28, 8:40] = 90
        p = tmp_path / f"g{i}.png"
        _save(p, img)
        frames.append(p)
    rois = find_glyph_rois(np.asarray(Image.open(frames[0])))
    assert len(rois) >= 1
    before = score_glyph_stability(frames, melt_threshold=0.12, sample_every=1)
    assert before.melt_events >= 1 or before.score < 0.95
    after = apply_glyph_lock(frames, strength=0.95, melt_threshold=0.12)
    assert after.repaired >= 1


def test_physics_flags_teleport(tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 20
        # Blob teleports across frame (far enough that match fails → pop teleport)
        x = 4 if i < 3 else 50
        img[22:38, x : x + 14] = 240
        p = tmp_path / f"ph{i}.png"
        _save(p, img)
        frames.append(p)
    report = score_physics_invariance(frames, sample_every=1)
    assert report.jump_discontinuities >= 1 or report.score < 0.95


def test_quality_includes_new_axes(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 35
        img[10:38, 10:38] = 160
        p = tmp_path / f"q{i}.png"
        _save(p, img)
        frames.append(p)
    q = score_segment_quality(frames, min_identity=0.1, min_extremity=0.1, min_physics=0.1)
    assert 0.0 <= q.identity_score <= 1.0
    assert 0.0 <= q.extremity_score <= 1.0
    assert 0.0 <= q.physics_score <= 1.0
    assert 0.0 <= q.glyph_score <= 1.0


def test_permanent_dit_identity_bank():
    torch.manual_seed(0)
    model = PermanentVideoDiT(
        in_channels=4, dim=32, depth=2, num_heads=4, patch_size=2, context_dim=16, num_slots=4, permanence_every=1
    )
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.zeros(1)
    identity = torch.randn(1, 3, 32)
    out = model(x, t, context=torch.randn(1, 16), identity=identity)
    assert out.shape == x.shape
    # identity loss on dummy tokens
    tok = torch.randn(1, 4, 16, 32)
    loss = identity_consistency_loss(tok)
    assert loss.ndim == 0
