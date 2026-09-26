import numpy as np
from utils.quality.color_neutrality import estimate_color_cast, neutralize_color_cast


def _warm_image(shape=(64, 64, 3)) -> np.ndarray:
    # Midtone gray with a strong warm (yellow) cast: R,G lifted, B suppressed.
    img = np.full(shape, 128.0, dtype=np.float32)
    img[..., 0] += 30
    img[..., 1] += 20
    img[..., 2] -= 35
    return np.clip(img, 0, 255).astype(np.uint8)


def _neutral_image(shape=(64, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(90, 170, size=shape, dtype=np.uint8).astype(np.uint8)


def test_estimate_detects_warm_cast():
    est = estimate_color_cast(_warm_image())
    assert est.magnitude > 0.06
    assert est.warm_bias > 0.1


def test_estimate_neutral_image_is_below_threshold():
    est = estimate_color_cast(_neutral_image())
    assert est.magnitude < 0.06
    assert abs(est.warm_bias) < 0.06


def test_neutralize_reduces_cast_magnitude():
    img = _warm_image()
    before = estimate_color_cast(img).magnitude
    out = neutralize_color_cast(img, strength=1.0)
    after = estimate_color_cast(out).magnitude
    assert after < before
    assert out.dtype == np.uint8


def test_neutralize_leaves_neutral_image_untouched():
    img = _neutral_image()
    out = neutralize_color_cast(img, strength=1.0)
    assert out is img  # threshold gate: no copy, no change


def test_warm_only_skips_cool_cast():
    img = _warm_image()[..., ::-1].copy()  # swap R/B -> cool cast
    out = neutralize_color_cast(img, strength=1.0, warm_only=True)
    assert out is img


def test_extreme_images_are_safe():
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    assert estimate_color_cast(black).magnitude == 0.0
    assert estimate_color_cast(white).magnitude == 0.0
    assert neutralize_color_cast(black, strength=1.0) is black
