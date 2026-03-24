"""Tests for face-region post-process helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_blend_reference_rgb_alpha_zero() -> None:
    from utils.quality.face_region_enhance import blend_reference_rgb

    base = np.zeros((8, 8, 3), dtype=np.uint8)
    ref = np.ones((4, 4, 3), dtype=np.uint8) * 200
    out = blend_reference_rgb(base, ref, alpha=0.0)
    assert out.shape == base.shape
    assert np.allclose(out, base)


def test_blend_reference_rgb_resizes() -> None:
    from utils.quality.face_region_enhance import blend_reference_rgb

    base = np.zeros((16, 16, 3), dtype=np.uint8)
    ref = np.ones((32, 32, 3), dtype=np.uint8) * 255
    out = blend_reference_rgb(base, ref, alpha=0.5)
    assert out.shape == (16, 16, 3)
    assert out.mean() > 100


def test_enhance_faces_no_opencv_or_no_faces_returns_uint8() -> None:
    from utils.quality.face_region_enhance import enhance_faces_in_rgb

    rng = np.random.default_rng(0)
    noise = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    out = enhance_faces_in_rgb(noise, max_faces=2)
    assert out.shape == noise.shape
    assert out.dtype == np.uint8
