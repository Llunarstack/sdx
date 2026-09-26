"""Human-art fingerprint scores, finish, and DiT lattice residuals."""

from __future__ import annotations

import numpy as np
import torch
from models.anti_ai_naturalness import AntiAINaturalnessController, LatticeBreakModule, detect_medium
from utils.quality.human_likeness import (
    apply_human_likeness_finish,
    score_bilateral_life,
    score_human_likeness,
    score_lattice_free,
    score_not_teal_orange,
    score_specular_life,
)
from utils.quality.human_made import human_made_prompt_fragments
from utils.quality.test_time_pick import _maybe_human_scores, _maybe_spatial_bind_scores


def _bars(period: int = 8, size: int = 128) -> np.ndarray:
    x = np.arange(size, dtype=np.uint8)
    col = ((x % period) < (period // 2)).astype(np.uint8) * 255
    g = np.broadcast_to(col[None, :], (size, size)).copy()
    return np.stack([g, g, g], axis=-1)


def test_mirror_scores_lower_than_lived_in() -> None:
    rng = np.random.default_rng(0)
    left = rng.integers(40, 220, (64, 32, 3), dtype=np.uint8)
    doll = np.concatenate([left, np.fliplr(left)], axis=1)
    life = rng.integers(40, 220, (64, 64, 3), dtype=np.uint8)
    assert score_bilateral_life(life) > score_bilateral_life(doll)
    assert score_human_likeness(life) > score_human_likeness(doll)


def test_teal_orange_lut_is_penalized() -> None:
    lut = np.zeros((64, 64, 3), dtype=np.uint8)
    lut[:32] = (20, 170, 215)
    lut[32:] = (95, 45, 12)
    plain = np.full((64, 64, 3), 128, dtype=np.uint8)
    plain[:32, :, 1] = 140
    assert score_not_teal_orange(plain) > score_not_teal_orange(lut)


def test_lattice_free_prefers_irregular_texture() -> None:
    rng = np.random.default_rng(1)
    irregular = rng.integers(30, 230, (128, 128, 3), dtype=np.uint8)
    tiled = _bars(8, 128)
    assert score_lattice_free(irregular) > score_lattice_free(tiled)


def test_specular_clone_is_penalized() -> None:
    doll = np.zeros((64, 64, 3), dtype=np.uint8)
    doll[:, :] = 40
    doll[20:24, 18:22] = 255
    doll[20:24, 42:46] = 255
    one = doll.copy()
    one[20:24, 42:46] = 40
    one[38:44, 10:16] = 255
    assert score_specular_life(one) > score_specular_life(doll)


def test_finish_is_identity_at_zero_and_changes_pixels() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    assert np.array_equal(apply_human_likeness_finish(img, strength=0.0, seed=7), img)
    out = apply_human_likeness_finish(img, strength=0.7, seed=7)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_strong_prompt_fragments_name_human_likeness() -> None:
    pos, neg = human_made_prompt_fragments("strong")
    blob = f"{pos} {neg}".lower()
    assert "catchlights" in blob
    assert "teal" in blob


def test_maybe_spatial_bind_not_shadowed_by_human_scores() -> None:
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    human = _maybe_human_scores("a portrait", imgs)
    bind = _maybe_spatial_bind_scores("a portrait", imgs)
    assert human is not None and len(human) == 2
    assert all(0.0 <= v <= 1.0 for v in human)
    # Portrait has no spatial color-bind, so the bind helper stays None.
    assert bind is None


def test_lattice_break_and_controller_change_tokens() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 64, 32)
    brk = LatticeBreakModule()
    y = brk(x, 8, 8, strength=0.8)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)
    ctrl = AntiAINaturalnessController(32)
    medium = detect_medium("photoreal dslr portrait photo")
    out = ctrl(x, medium, 8, 8, strength=0.5)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)
