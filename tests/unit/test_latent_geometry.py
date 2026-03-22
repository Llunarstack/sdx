"""Tests for :mod:`sdx_native.latent_geometry` (matches native C++ latent helpers)."""

from utils.latent_geometry import (
    dit_patch_size_from_variant_name,
    latent_spatial_size,
    num_patch_tokens,
    patch_grid_dim,
)


def test_latent_spatial_size_divisible():
    assert latent_spatial_size(256, 8) == 32
    assert latent_spatial_size(256, 7) == 0


def test_num_patch_tokens():
    # 32x32 latent, patch 2 -> 16x16 = 256 tokens
    assert num_patch_tokens(256, 8, 2) == 256
    assert num_patch_tokens(256, 8, 4) == 64


def test_patch_grid_dim():
    assert patch_grid_dim(32, 2) == 16


def test_dit_patch_size_from_name():
    assert dit_patch_size_from_variant_name("DiT-XL/2-Text") == 2
    assert dit_patch_size_from_variant_name("DiT-B/4-Text") == 4
