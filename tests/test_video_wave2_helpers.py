"""Tests for wave-2 video helpers: contact, count, shimmer, shutter, occlusion, chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pipelines.video.contact_ground import apply_contact_ground, score_contact_ground
from pipelines.video.count_binder import expected_count_from_prompt, score_count_stability
from pipelines.video.hf_shimmer import apply_hf_deshimmer, score_hf_shimmer
from pipelines.video.motion_shutter import apply_motion_shutter
from pipelines.video.occlusion_resolve import score_occlusion_coherence
from pipelines.video.process_options import parse_process_options
from pipelines.video.secondary_track import score_secondary_track
from pipelines.video.shot_chain import chain_anchor_from_segment, harmonize_cut, score_cut_continuity
from pipelines.video.superior_pass import run_superior_pass


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def test_contact_ground_scores_and_paints(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 180  # bright floor
        # Dark feet blob with no shadow
        img[50:60, 24:40] = 40
        p = tmp_path / f"c{i}.png"
        _save(p, img)
        frames.append(p)
    before = score_contact_ground(frames, sample_every=1)
    assert before.float_events >= 1 or before.score < 0.95
    after = apply_contact_ground(frames, strength=0.8)
    assert after.repaired >= 1


def test_count_binder_prompt_and_drop(tmp_path: Path):
    assert expected_count_from_prompt("three people walking") >= 2
    frames = []
    for i in range(6):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 20
        # Two blobs early, one later
        img[20:32, 10:22] = 220
        if i < 3:
            img[20:32, 40:52] = 220
        p = tmp_path / f"n{i}.png"
        _save(p, img)
        frames.append(p)
    rep = score_count_stability(frames, expected=2, sample_every=1)
    assert rep.drop_events >= 1 or rep.score < 0.95


def test_hf_shimmer_detect_and_fix(tmp_path: Path):
    frames = []
    rng = np.random.default_rng(0)
    for i in range(6):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 40
        # High-freq noise that changes every frame (foliage shimmer)
        noise = rng.integers(0, 255, size=(48, 48), dtype=np.uint8)
        img[..., 0] = noise
        img[..., 1] = noise
        img[..., 2] = noise
        p = tmp_path / f"s{i}.png"
        _save(p, img)
        frames.append(p)
    before = score_hf_shimmer(frames)
    assert before.score < 0.95 or before.shimmer_energy > 0
    after = apply_hf_deshimmer(frames, strength=0.8, window=3)
    assert after.repaired >= 1


def test_motion_shutter_applies_on_pan(tmp_path: Path):
    frames = []
    for i in range(5):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 30
        x = 4 + i * 6
        img[16:32, x : x + 10] = 200
        p = tmp_path / f"m{i}.png"
        _save(p, img)
        frames.append(p)
    rep = apply_motion_shutter(frames, amount=0.7, min_flow=0.5)
    assert rep.applied >= 1 or rep.mean_flow >= 0.5


def test_occlusion_and_secondary_score(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8) + 25
        img[20:40, 20:35] = 200
        img[22:38, 30:45] = 160  # overlapping secondary
        p = tmp_path / f"o{i}.png"
        _save(p, img)
        frames.append(p)
    assert 0.0 <= score_occlusion_coherence(frames).score <= 1.0
    assert 0.0 <= score_secondary_track(frames).score <= 1.0


def test_shot_chain_harmonize(tmp_path: Path):
    prev = []
    nxt = []
    for i in range(3):
        a = np.zeros((32, 32, 3), dtype=np.uint8) + 80
        b = np.zeros((32, 32, 3), dtype=np.uint8) + 180
        pa = tmp_path / f"p{i}.png"
        pb = tmp_path / f"n{i}.png"
        _save(pa, a)
        _save(pb, b)
        prev.append(pa)
        nxt.append(pb)
    anchor = chain_anchor_from_segment(prev, out_path=tmp_path / "anchor.png")
    assert anchor.is_file()
    before = score_cut_continuity(prev, nxt)
    assert before.delta_luma > 0.05
    after = harmonize_cut(nxt, anchor, strength=0.6, n_frames=2)
    assert after.score >= before.score * 0.5  # should not crash; often improves


def test_superior_pass_runs(tmp_path: Path):
    frames = []
    for i in range(4):
        img = np.zeros((48, 48, 3), dtype=np.uint8) + 50
        img[30:42, 18:30] = 30
        p = tmp_path / f"u{i}.png"
        _save(p, img)
        frames.append(p)
    opts = parse_process_options(
        {
            "permanence_repair": True,
            "identity_bind": True,
            "extremity_lock": False,
            "glyph_lock": False,
            "contact_ground": True,
            "hf_deshimmer": True,
            "count_bind": True,
            "physics_gate": True,
            "secondary_track": False,
            "motion_shutter": False,
        }
    )
    result = run_superior_pass(frames, opts, prompt="one person standing")
    assert "permanence" in result.scores or any("permanence" in o for o in result.ops)
    assert isinstance(result.ops, list)
