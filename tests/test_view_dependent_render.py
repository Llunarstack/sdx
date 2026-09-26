"""Tests for wiring SH view-dependent radiance into the NeuS renderer (3D)."""

from __future__ import annotations

import torch
from frontier.multiview import (
    NeusRenderer,
    SHRadianceDecoder,
    SpatialLatentField,
    TriPlaneConfig,
    look_at_rays,
)

CFG = TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16)


def _setup(sh: bool):
    torch.manual_seed(0)
    field = SpatialLatentField(CFG)
    decoder = SHRadianceDecoder(feature_dim=CFG.feature_dim, degree=3) if sh else None
    renderer = NeusRenderer(n_samples=24, sh_decoder=decoder)
    planes = field.encode(torch.zeros(1, CFG.cond_dim))
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), height=6, width=6)
    return field, renderer, planes, o, d


def test_sh_renderer_runs_and_is_bounded():
    field, renderer, planes, o, d = _setup(sh=True)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert out["rgb"].shape == (36, 3)
    assert torch.isfinite(out["rgb"]).all()
    assert torch.all((out["rgb"] >= 0) & (out["rgb"] <= 1))


def test_sh_decoder_receives_gradient():
    field, renderer, planes, o, d = _setup(sh=True)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    out["rgb"].mean().backward()
    assert any(p.grad is not None for p in renderer.sh_decoder.parameters())


def test_sh_path_differs_from_flat_albedo():
    # Same field + camera, with and without the SH decoder, should differ:
    # confirms the view-dependent path is actually engaged.
    field, renderer_sh, planes, o, d = _setup(sh=True)
    renderer_flat = NeusRenderer(n_samples=24)  # no sh_decoder
    rgb_sh = renderer_sh.render(field, planes, o, d, near=0.5, far=3.5)["rgb"]
    rgb_flat = renderer_flat.render(field, planes, o, d, near=0.5, far=3.5)["rgb"]
    assert not torch.allclose(rgb_sh, rgb_flat)


def test_flat_renderer_unchanged_by_default():
    # Default renderer (no decoder) still returns valid albedo colors.
    field, renderer, planes, o, d = _setup(sh=False)
    assert renderer.sh_decoder is None
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert torch.all((out["rgb"] >= 0) & (out["rgb"] <= 1))
