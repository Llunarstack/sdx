"""Tests for utils.ai_accel (NumPy fallbacks always; native optional).

Also holds what used to be: test_ai_accel, test_ai_image_shortcomings, test_ai_look.
"""

from __future__ import annotations

import cv2
import numpy as np
from config.defaults._lexicon import merge_csv_unique
from config.defaults.ai_image_shortcomings import detect_shortcoming_ids, mitigation_fragments, spec_by_id
from data.caption_utils import apply_training_guidance_to_caption_pair
from utils.ai_accel import native_stack_summary, quality_highlight_frac
from utils.quality import ai_look
from utils.quality.camera_isp import linear_to_srgb, srgb_to_linear

# ==================================================================================================
# test_ai_accel: Tests for utils.ai_accel (NumPy fallbacks always; native optional).
# ==================================================================================================


def test_quality_highlight_frac_matches_numpy():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:2] = 255
    assert abs(quality_highlight_frac(rgb, thr=250.0) - 0.25) < 1e-6


def test_native_backends_built_when_present():
    """If the Rust DLL exists from build_native, the wrapper must report available=True."""
    from sdx_native import quality_scorers_native
    from sdx_native.native_tools import shared_library_path

    if shared_library_path("sdx_quality_scorers"):
        assert quality_scorers_native.available()
    assert "paths" in native_stack_summary()


# ==================================================================================================
# test_ai_image_shortcomings: Tests for config.defaults.ai_image_shortcomings (prompt mitigation packs).
# ==================================================================================================


def test_merge_csv_unique_dedupes():
    assert merge_csv_unique("a, b", "b, c") == "a, b, c"


def test_detect_portrait_skin():
    ids = detect_shortcoming_ids("portrait, detailed face, natural skin", include_2d=False)
    assert "skin_detail_tangents" in ids


def test_detect_digital_painting_keywords():
    ids = detect_shortcoming_ids("warrior, digital painting, procreate, detailed", include_2d=False)
    assert "digital_painting" in ids


def test_detect_pixel_art():
    ids = detect_shortcoming_ids("16-bit pixel art sprite, rpg enemy", include_2d=False)
    assert "pixel_digital" in ids


def test_mitigation_auto_digital_art():
    pos, neg = mitigation_fragments(
        "concept art environment, matte painting, artstation", "auto", include_2d_pack=False
    )
    assert "design" in pos.lower() or "perspective" in pos.lower()
    assert "cutout" in neg.lower() or "mismatch" in neg.lower()


def test_detect_anime_requires_2d_flag():
    ids_no = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=False)
    assert "style_drift_2d" not in ids_no
    ids_yes = detect_shortcoming_ids("1girl, anime style, cel shaded", include_2d=True)
    assert "style_drift_2d" in ids_yes


def test_mitigation_auto_non_empty_for_matched():
    pos, neg = mitigation_fragments("woman sitting on a chair in sunlight", "auto", include_2d_pack=False)
    assert pos
    assert neg
    assert "floating" in neg.lower() or "contradictory" in neg.lower()


def test_mitigation_all_includes_many_hints():
    pos, neg = mitigation_fragments("", "all", include_2d_pack=False)
    assert len(pos) > 80
    assert len(neg) > 80


def test_spec_by_id():
    assert spec_by_id("lighting_gi").id == "lighting_gi"


def test_training_guidance_shortcomings_none_passthrough():
    c, n = apply_training_guidance_to_caption_pair("hello", "bad", shortcomings_mode="none")
    assert c == "hello"
    assert n == "bad"


def test_training_guidance_shortcomings_auto_merges_negative():
    c, n = apply_training_guidance_to_caption_pair(
        "portrait, woman sitting in sunlight", "blurry", shortcomings_mode="auto"
    )
    assert "blurry" in n
    assert "floating" in n.lower() or "contradictory" in n.lower()
    assert len(c) > len("portrait, woman sitting in sunlight")


# ==================================================================================================
# test_ai_look: AI-look critic on synthetic images: each detector separates a tell from its natural counterpart.
# ==================================================================================================


def _pink(n: int = 256, seed: int = 0) -> np.ndarray:
    """1/f^2 power-spectrum field (natural-image statistics), float, mean 128 std 40."""
    rng = np.random.default_rng(seed)
    f = np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :])
    f[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f
    spec[0, 0] = 0.0
    x = np.real(np.fft.ifft2(spec))
    return (x - x.mean()) / x.std() * 40.0 + 128.0


def _rgb(gray: np.ndarray) -> np.ndarray:
    u = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([u, u, u], axis=-1)


def test_decoder_grid_flags_nearest_upsampling_not_one_over_f() -> None:
    small = np.random.default_rng(1).normal(128, 40, (32, 32))
    upsampled = _rgb(np.kron(small, np.ones((8, 8))))
    assert ai_look.score_decoder_grid(upsampled) == 0.0
    for seed in range(3):
        assert ai_look.score_decoder_grid(_rgb(_pink(seed=seed))) == 1.0


def test_decoder_grid_flags_a_faint_fingerprint() -> None:
    pattern = np.random.default_rng(2).normal(0, 3.0, (16, 16))
    stamped = _pink(seed=4) + np.tile(pattern, (16, 16))
    assert ai_look.score_decoder_grid(_rgb(stamped)) < 0.2


def test_oversaturated_image_scores_worse() -> None:
    rgb = np.stack([np.clip(_pink(seed=s) * 0.6 + m, 0, 255) for s, m in ((5, 60), (6, 50), (7, 40))], -1)
    natural = rgb.astype(np.uint8)
    hsv = cv2.cvtColor(natural, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 3.0, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    assert ai_look.score_oversaturation(natural) > ai_look.score_oversaturation(boosted)
    assert ai_look.ai_look_score(boosted).ai_look > ai_look.ai_look_score(natural).ai_look
    # A requested vivid look relaxes the ramp.
    assert ai_look.score_oversaturation(boosted, vivid_ok=True) >= ai_look.score_oversaturation(boosted)


def test_noise_floor_prefers_signal_dependent_capture_noise() -> None:
    ramp = np.tile(np.linspace(20, 235, 256, dtype=np.float32), (256, 1))
    rng = np.random.default_rng(3)
    lin = srgb_to_linear(ramp / 255.0)
    captured = linear_to_srgb(np.clip(lin + rng.normal(size=lin.shape) * np.sqrt(0.004 * lin + 4e-6), 0, 1)) * 255
    grain = ramp + rng.normal(0, 2.0, ramp.shape)
    clean = ai_look.score_noise_floor(_rgb(ramp))
    flat = ai_look.score_noise_floor(_rgb(grain))
    real = ai_look.score_noise_floor(_rgb(captured))
    assert clean == 0.0
    assert real is not None and flat is not None and real > flat > clean


def test_blur_reads_as_oversmoothed_and_steep() -> None:
    sharp, waxy = _rgb(_pink(seed=8)), _rgb(cv2.GaussianBlur(_pink(seed=8), (0, 0), 1.5))
    assert ai_look.score_oversmoothing(sharp) > ai_look.score_oversmoothing(waxy)
    assert ai_look.score_spectral_slope(sharp) > ai_look.score_spectral_slope(waxy)


def test_halo_detects_unsharp_mask_overshoot() -> None:
    scene = np.full((192, 192), 70.0, np.float32)
    cv2.circle(scene, (96, 96), 50, 190, -1)
    scene = cv2.GaussianBlur(scene, (0, 0), 1.2)
    sharpened = scene + 1.5 * (scene - cv2.GaussianBlur(scene, (0, 0), 2.0))
    assert ai_look.score_halo(_rgb(scene)) == 1.0
    assert ai_look.score_halo(_rgb(sharpened)) < 0.5
    assert ai_look.score_halo(_rgb(_pink())) is None  # texture is not a step edge


def test_mirror_symmetry_penalised_unless_requested() -> None:
    img = np.stack([np.clip(_pink(128, seed=s), 0, 255) for s in (9, 10, 11)], -1).astype(np.uint8)
    mirrored = img.copy()
    mirrored[:, 64:] = img[:, :64][:, ::-1]
    assert ai_look.score_symmetry(img) > ai_look.score_symmetry(mirrored)
    assert "symmetry" not in ai_look.ai_look_score(mirrored, "a symmetrical mandala").parts


def test_drawn_media_skip_capture_detectors() -> None:
    parts = ai_look.ai_look_score(_rgb(_pink()), "anime girl, cel shading").parts
    assert not set(parts) & {"noise_floor", "spectral_slope", "oversmoothing"}
    assert "grid" not in ai_look.ai_look_score(_rgb(_pink()), "pixel art castle").parts
