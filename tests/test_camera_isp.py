"""Capture finish: lens, Poisson-Gaussian sensor, ISP, film grain, medium routing, output quantisation."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest
from utils.quality.camera_isp import (
    CAMERA_PRESETS,
    NO_LENS,
    CameraProfile,
    FilmStock,
    Lens,
    apply_finish,
    demosaic_malvar,
    lateral_chromatic_aberration,
    mosaic_rggb,
    plan_finish,
    poisson_gaussian_noise,
    save_rgb,
    simulate_capture,
    simulate_film,
    srgb_to_linear,
    to_uint8,
    vignette_gain,
)

NOISELESS = CameraProfile(
    "noiseless",
    lens=NO_LENS,
    full_well_e=1e12,
    read_noise_e=0.0,
    post_read_noise_e=0.0,
    cfa=False,
    tone=0.0,
    sharpen=0.0,
    jpeg_quality=0,
)


def _ramp(h: int = 64, w: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.stack([40 + 150 * xx / w, 60 + 120 * yy / h, 200 - 120 * xx / w], axis=-1)


def test_sensor_noise_variance_grows_linearly_with_signal() -> None:
    rng = np.random.default_rng(0)
    levels = np.array([0.02, 0.1, 0.3, 0.7])
    var = [
        poisson_gaussian_noise(np.full((256, 256), m), full_well_e=5000, read_noise_e=3, rng=rng).var() for m in levels
    ]
    slope, intercept = np.polyfit(levels, var, 1)
    assert np.all(np.diff(var) > 0)
    assert slope == pytest.approx(1 / 5000, rel=0.1)
    assert intercept == pytest.approx((3 / 5000) ** 2, abs=2e-6)


def test_noise_grows_with_iso() -> None:
    flat = np.full((64, 64, 3), 120.0, np.float32)
    base = replace(CAMERA_PRESETS["dslr"], lens=NO_LENS, sharpen=0.0, chroma_nr=0.0)
    low = simulate_capture(flat, replace(base, iso=100), seed=1, jpeg=False).std()
    high = simulate_capture(flat, replace(base, iso=3200), seed=1, jpeg=False).std()
    assert high > 3 * low > 0


def test_unprocess_reprocess_is_identity_without_effects() -> None:
    img = _ramp()
    out = simulate_capture(img, NOISELESS, jpeg=False)
    assert np.abs(out - img).max() < 0.05


def test_malvar_demosaic_reconstructs_flat_colour() -> None:
    rgb = np.broadcast_to(np.array([0.2, 0.5, 0.8], np.float32), (16, 16, 3))
    assert np.allclose(demosaic_malvar(mosaic_rggb(rgb)), rgb, atol=1e-5)


def _centroid(plane: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[0 : plane.shape[0], 0 : plane.shape[1]]
    w = plane / plane.sum()
    return np.array([(w * ys).sum(), (w * xs).sum()])


def test_chromatic_aberration_is_radial_and_fixes_the_centre() -> None:
    h = w = 129
    centre, mid, far = (64, 64), (64 + 20, 64 + 15), (64 + 48, 64 + 36)
    shifts = []
    for y, x in (centre, mid, far):
        img = np.zeros((h, w, 3), np.float32)
        img[y - 1 : y + 2, x - 1 : x + 2, :] = 1.0
        out = lateral_chromatic_aberration(img, 0.02, -0.02)
        assert np.array_equal(out[..., 1], img[..., 1])
        red = _centroid(np.clip(out[..., 0], 0, None)) - (y, x)
        blue = _centroid(np.clip(out[..., 2], 0, None)) - (y, x)
        shifts.append((red, blue, np.array([y - 64, x - 64], np.float64)))
    assert np.abs(shifts[0][0]).max() < 1e-4 and np.abs(shifts[0][1]).max() < 1e-4
    for red, blue, radial in shifts[1:]:
        unit = radial / np.linalg.norm(radial)
        assert red @ unit > 0 > blue @ unit
        assert abs(red[0] * unit[1] - red[1] * unit[0]) < 0.02 * np.linalg.norm(red)
    assert np.linalg.norm(shifts[2][0]) > np.linalg.norm(shifts[1][0])


def test_vignette_is_applied_exactly_once() -> None:
    flat = np.full((65, 97, 3), 150.0, np.float32)
    lens = Lens(vignette=0.6, ca_red=0.0, ca_blue=0.0, psf_sigma=0.0)
    out = simulate_capture(flat, replace(NOISELESS, lens=lens), jpeg=False)
    measured = srgb_to_linear(out / 255.0)[..., 1] / srgb_to_linear(np.float32(150 / 255))
    assert np.allclose(measured, vignette_gain(65, 97, lens), atol=2e-3)
    assert measured[32, 48] == pytest.approx(1.0, abs=1e-3)


def test_finish_has_no_seam_or_wraparound_ghost() -> None:
    img = np.zeros((64, 96, 3), np.float32)
    img[:, :8] = 255.0
    plan = replace(CAMERA_PRESETS["dslr"], iso=100)
    out = apply_finish(img, plan, seed=3).astype(np.float32)
    assert out[:, -8:].mean() < 2.0
    ramp = apply_finish(_ramp(), plan, seed=3).astype(np.float32)
    col_step = np.abs(np.diff(ramp.mean(axis=0), axis=0)).max(axis=-1)
    assert col_step[44:52].max() < 3 * np.median(col_step) + 1.0


def test_seed_determinism() -> None:
    img = _ramp()
    plan = plan_finish("street photo at night")
    a = apply_finish(img, plan, seed=5)
    assert np.array_equal(a, apply_finish(img, plan, seed=5))
    assert not np.array_equal(a, apply_finish(img, plan, seed=6))
    film = plan_finish("portra 400 film photo")
    assert np.array_equal(apply_finish(img, film, seed=2), apply_finish(img, film, seed=2))


def test_medium_routing() -> None:
    for prompt in ("anime girl, cel shading", "watercolor painting of a harbor", "octane 3d render", "pixel art"):
        assert plan_finish(prompt) is None
    assert plan_finish("anything", camera="none") is None
    assert isinstance(plan_finish("portrait photo, 85mm"), CameraProfile)
    assert plan_finish("candid iphone photo").name == "phone"
    assert plan_finish("portrait photo").iso < plan_finish("portrait photo at night").iso
    assert plan_finish("photo, ISO 3200").iso == 3200
    film = plan_finish("black and white film photo, tri-x")
    assert isinstance(film, FilmStock) and film.monochrome
    assert isinstance(plan_finish("a cafe", grain_style="heavy_16mm"), FilmStock)


def test_anime_route_adds_no_noise() -> None:
    flat = np.full((32, 32, 3), 123.4, np.float32)
    out = to_uint8(apply_finish(flat, plan_finish("anime girl, cel shading"), seed=1))
    assert np.array_equal(out, np.full_like(out, 123))


def test_film_grain_is_signal_dependent_without_sensor_noise() -> None:
    stock = FilmStock("t", iso=3200, lens=NO_LENS, frame_width_mm=10.0)
    stds = []
    for level in (30.0, 120.0, 230.0):
        out = simulate_film(np.full((96, 96, 3), level, np.float32), stock, seed=0)
        assert out.mean() == pytest.approx(level, abs=0.5)
        stds.append(float(out.std()))
    assert max(stds) > 1.3 * min(stds) > 0
    mono = simulate_film(_ramp(), replace(stock, monochrome=True), seed=0)
    gain = srgb_to_linear(mono / 255.0) / srgb_to_linear(_ramp() / 255.0)
    assert np.allclose(gain[..., 0], gain[..., 1], atol=2e-3)


def test_dithered_quantisation_and_16bit_png(tmp_path) -> None:
    flat = np.full((64, 64, 3), 100.3, np.float32)
    assert to_uint8(flat).mean() == 100
    assert to_uint8(flat, np.random.default_rng(0)).mean() == pytest.approx(100.3, abs=0.05)
    path = tmp_path / "out.png"
    save_rgb(flat, path, sixteen_bit=True)
    back = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16 and abs(int(back[0, 0, 0]) - round(100.3 * 257)) <= 1
