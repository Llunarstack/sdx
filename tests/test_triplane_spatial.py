"""Tests for the tri-plane spatial latent field (SDX-Spatial 3D core).

The load-bearing property is multi-view consistency: a point in object space
must decode to the same geometry/material regardless of which camera asked.
"""

from __future__ import annotations

import pytest
import torch
from frontier.multiview import SpatialLatentField, TriPlaneConfig, build_spatial_field

# Small config so the whole suite runs fast on CPU.
CFG = TriPlaneConfig(
    cond_dim=32,
    plane_channels=8,
    grid_res=16,
    base_channels=32,
    feature_dim=16,
    decoder_hidden=32,
)


@pytest.fixture
def field() -> SpatialLatentField:
    torch.manual_seed(0)
    return SpatialLatentField(CFG)


def _planes_and_points(field: SpatialLatentField, b: int = 2, n: int = 64):
    cond = torch.randn(b, CFG.cond_dim)
    planes = field.encode(cond)
    points = torch.empty(b, n, 3).uniform_(-1.0, 1.0)
    return planes, points


def test_encode_plane_shapes(field: SpatialLatentField) -> None:
    planes, _ = _planes_and_points(field)
    assert len(planes) == 3
    for p in planes:
        assert p.shape == (2, CFG.plane_channels, CFG.grid_res, CFG.grid_res)


def test_query_feature_shape(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    feat = field.query(planes, points)
    assert feat.shape == (2, 64, CFG.feature_dim)


def test_splat_decoder_ranges_and_shapes(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    s = field.decode_splats(planes, points)
    assert s.positions.shape == (2, 64, 3)
    assert s.scales.shape == (2, 64, 3)
    assert torch.all(s.scales > 0)
    assert torch.all((s.opacities >= 0) & (s.opacities <= 1))
    assert torch.all((s.colors >= 0) & (s.colors <= 1))
    # rotations are unit quaternions
    norms = s.rotations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # splat position stays near its query point (bounded offset)
    assert torch.all((s.positions - points).abs() <= 0.1 + 1e-4)


def test_sdf_shape(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    sdf = field.decode_sdf(planes, points)
    assert sdf.shape == (2, 64, 1)
    assert torch.isfinite(sdf).all()


def test_pbr_ranges(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field)
    pbr = field.decode_pbr(planes, points)
    assert torch.all((pbr.albedo >= 0) & (pbr.albedo <= 1))
    assert torch.all((pbr.roughness >= 0) & (pbr.roughness <= 1))
    assert torch.all((pbr.metallic >= 0) & (pbr.metallic <= 1))
    normal_norms = pbr.normal.norm(dim=-1)
    assert torch.allclose(normal_norms, torch.ones_like(normal_norms), atol=1e-5)


def test_multiview_consistency_is_view_independent(field: SpatialLatentField) -> None:
    """The core guarantee: same object-space point -> identical surface, no
    matter the query order or how the 'camera' reached it."""
    field.eval()
    cond = torch.randn(1, CFG.cond_dim)
    planes = field.encode(cond)
    pts = torch.empty(1, 32, 3).uniform_(-0.9, 0.9)

    # "Front camera" queries the points directly.
    front = field.decode_pbr(planes, pts)
    # "Side camera" reaches the same points in a shuffled order.
    perm = torch.randperm(32)
    side = field.decode_pbr(planes, pts[:, perm])

    assert torch.allclose(front.albedo[:, perm], side.albedo, atol=1e-6)
    assert torch.allclose(front.roughness[:, perm], side.roughness, atol=1e-6)


def test_different_conditioning_gives_different_field(field: SpatialLatentField) -> None:
    a = field.encode(torch.randn(1, CFG.cond_dim))
    b = field.encode(torch.randn(1, CFG.cond_dim))
    assert not torch.allclose(a[0], b[0])


def test_render_view_runs(field: SpatialLatentField) -> None:
    planes = field.encode(torch.randn(1, CFG.cond_dim))
    cam_origin = torch.tensor([[0.0, 0.0, -2.0]])
    # A tiny 4x4 pinhole ray bundle pointing down +z.
    ys, xs = torch.meshgrid(torch.linspace(-0.3, 0.3, 4), torch.linspace(-0.3, 0.3, 4), indexing="ij")
    dirs = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(16)), dim=-1)
    dirs = torch.nn.functional.normalize(dirs, dim=-1).unsqueeze(0)
    rgb, depth, hit = field.render_view(planes, cam_origin, dirs, steps=8)
    assert rgb.shape == (1, 16, 3)
    assert depth.shape == (1, 16)
    assert hit.shape == (1, 16)
    assert torch.isfinite(rgb).all()
    assert torch.all((rgb >= 0) & (rgb <= 1))


def test_gradients_flow_through_all_heads(field: SpatialLatentField) -> None:
    planes, points = _planes_and_points(field, b=1, n=16)
    sdf = field.decode_sdf(planes, points)
    pbr = field.decode_pbr(planes, points)
    splats = field.decode_splats(planes, points)
    loss = sdf.mean() + pbr.albedo.mean() + splats.opacities.mean()
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_builder_helper() -> None:
    f = build_spatial_field(cond_dim=16, grid_res=8, plane_channels=4, feature_dim=8)
    assert f.cfg.cond_dim == 16
    assert f.cfg.grid_res == 8


def test_invalid_grid_res_rejected() -> None:
    with pytest.raises(ValueError):
        TriPlaneConfig(grid_res=24)  # not a power of two
