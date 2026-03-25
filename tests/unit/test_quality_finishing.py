"""Cross-style post-process helpers: monotonicity, bounds, presets."""

import numpy as np
import pytest

from utils.quality.quality import (
    FINISHING_PRESET_BASELINES,
    chroma_smooth_light,
    gentle_s_curve_luminance,
    luminance_clarity,
    polish_pass,
)


def test_finishing_presets_keys():
    for k in ("none", "photo", "anime", "illustration", "characters", "painterly"):
        assert k in FINISHING_PRESET_BASELINES
        assert len(FINISHING_PRESET_BASELINES[k]) == 3


def test_gentle_s_curve_luminance_bounds():
    rng = np.random.default_rng(0)
    img = (rng.random((24, 32, 3)) * 255).astype(np.float32)
    out = gentle_s_curve_luminance(img, strength=0.25)
    assert out.shape == img.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 255.0


@pytest.mark.parametrize("fn", [chroma_smooth_light, luminance_clarity])
def test_scipy_ops_identity_at_zero(fn):
    img = np.zeros((16, 16, 3), dtype=np.uint8) + 128
    assert np.array_equal(fn(img, 0.0), img)


def test_polish_pass_off():
    img = np.ones((8, 8, 3), dtype=np.uint8) * 200
    assert np.array_equal(polish_pass(img, 0.0), img)


def test_polish_pass_bounded():
    rng = np.random.default_rng(1)
    img = (rng.random((20, 20, 3)) * 255).astype(np.uint8)
    out = polish_pass(img, 0.5, seed=42)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255
