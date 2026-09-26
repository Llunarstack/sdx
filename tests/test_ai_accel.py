"""Tests for utils.ai_accel (NumPy fallbacks always; native optional)."""

from __future__ import annotations

import numpy as np
from utils.ai_accel import (
    canny_control_map,
    cfg_combine,
    quality_highlight_frac,
    quality_laplacian_var,
    style_l2_normalize,
    style_subtract,
    style_weighted_mean,
)


def test_style_weighted_mean_and_subtract():
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    w = np.array([0.75, 0.25], dtype=np.float32)
    mean = style_weighted_mean(rows, w)
    assert mean.shape == (3,)
    assert abs(float(mean[0]) - 0.75) < 1e-5
    assert abs(float(mean[1]) - 0.25) < 1e-5

    sub = style_subtract(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), strength=1.0)
    assert np.allclose(sub, np.array([0.5, 1.5, 2.5], dtype=np.float32))


def test_style_l2_normalize():
    v = style_l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_cfg_combine_classic():
    cond = np.ones(8, dtype=np.float32) * 2.0
    uncond = np.ones(8, dtype=np.float32)
    out = cfg_combine(cond, uncond, scale=2.0)
    assert np.allclose(out, np.ones(8, dtype=np.float32) * 3.0)


def test_canny_control_map_runs():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24] = 255
    edges = canny_control_map(rgb)
    assert edges.shape == (32, 32)
    assert edges.dtype == np.uint8
    assert int(edges.max()) > 0


def test_quality_kernels_smoke():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    assert quality_laplacian_var(rgb) >= 0.0
    assert 0.0 <= quality_highlight_frac(rgb, thr=250.0) <= 1.0


def test_native_backends_built_when_present():
    """If DLLs exist from build_native, wrappers must report available=True."""
    from sdx_native import canny_ops_native, cfg_combine_native, quality_scorers_native, style_embed_native
    from sdx_native.native_tools import (
        c_cfg_combine_shared_library_path,
        rust_canny_ops_shared_library_path,
        rust_quality_scorers_shared_library_path,
        rust_style_embed_shared_library_path,
    )

    if rust_canny_ops_shared_library_path():
        assert canny_ops_native.available()
    if rust_style_embed_shared_library_path():
        assert style_embed_native.available()
    if rust_quality_scorers_shared_library_path():
        assert quality_scorers_native.available()
    if c_cfg_combine_shared_library_path():
        assert cfg_combine_native.available()
