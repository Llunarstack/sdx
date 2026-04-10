"""Tests for ``diffusion.pixel_perfect``."""

from __future__ import annotations

import pytest
from diffusion import pixel_perfect as pp


def test_snap_to_multiple() -> None:
    assert pp.snap_to_multiple(100, 8, mode="floor") == 96
    assert pp.snap_to_multiple(100, 8, mode="ceil") == 104
    assert pp.snap_to_multiple(100, 8, mode="nearest") == 96


def test_latent_hw_roundtrip() -> None:
    assert pp.latent_hw_from_pixels(512, 512) == (64, 64)
    assert pp.pixels_from_latent_hw(64, 64) == (512, 512)


def test_pixel_stride_no_model() -> None:
    assert pp.pixel_stride_for_pipeline() == pp.LATENT_TO_PIXEL


def test_dit_rgb_stride_mock_model() -> None:
    class Emb:
        patch_size = 2

    class M:
        x_embedder = Emb()

    assert pp.dit_rgb_stride_px(M()) == 16


def test_resolve_with_dit_model_snaps_to_16() -> None:
    class Emb:
        patch_size = 2

    class M:
        x_embedder = Emb()

    h, w, spec = pp.resolve_pixel_perfect_hw(513, 500, model=M(), mode="nearest")
    assert h % 16 == 0 and w % 16 == 0
    assert spec.stride_px == 16
    assert spec.aligned_to_dit_patch is True
    assert spec.latent_h == h // 8


def test_resolve_square() -> None:
    h, w, spec = pp.resolve_pixel_perfect_hw(500, 510, square=True, mode="nearest")
    assert h == w
    assert h % 8 == 0


def test_validate_pixels_against_dit_ok() -> None:
    class Emb:
        img_size = 64

    class M:
        x_embedder = Emb()

    pp.validate_pixels_against_dit(512, 512, M())


def test_validate_pixels_against_dit_bad() -> None:
    class Emb:
        img_size = 64

    class M:
        x_embedder = Emb()

    with pytest.raises(ValueError, match="latent"):
        pp.validate_pixels_against_dit(500, 512, M())


def test_ar_block_grid_side() -> None:
    assert pp.ar_block_grid_side(16) == 4
    assert pp.ar_block_grid_side(15) is None


def test_validate_latent_matches_ar_grid() -> None:
    pp.validate_latent_matches_ar_grid(4, 4, 16)
    with pytest.raises(ValueError):
        pp.validate_latent_matches_ar_grid(4, 5, 16)


def test_tag_manifest_pixel_perfect() -> None:
    h, w, spec = pp.resolve_pixel_perfect_hw(256, 256, mode="nearest")
    row = pp.tag_manifest_pixel_perfect({"caption": "x"}, spec)
    assert row["height_px"] == h
    assert row["pixel_perfect_stride_px"] == spec.stride_px
