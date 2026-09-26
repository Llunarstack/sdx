"""Tests for NeuS-style volume rendering and photometric multi-view training.

The rendering math is validated against an *analytic* sphere (independent of the
network): a camera looking at a sphere must see a centered disk silhouette at
roughly the correct depth. Then the full path is checked: render a real field,
backprop, and overfit the field to posed images of a sphere.
"""

from __future__ import annotations

import numpy as np
import torch
from frontier.multiview import (
    NeusRenderer,
    SpatialLatentField,
    TriPlaneConfig,
    View,
    fit_from_views,
    look_at_rays,
    volume_render_sdf,
)

INV_S = torch.tensor(30.0)


def _sphere_scene(radius: float):
    def sdf_fn(p: torch.Tensor) -> torch.Tensor:
        return p.norm(dim=-1) - radius

    def color_fn(p: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.nn.functional.normalize(p, dim=-1)  # normal-as-color

    return sdf_fn, color_fn


def test_look_at_rays_shapes_and_unit():
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), height=8, width=8)
    assert o.shape == (64, 3) and d.shape == (64, 3)
    assert torch.allclose(d.norm(dim=-1), torch.ones(64), atol=1e-5)


def test_analytic_sphere_silhouette_is_a_centered_disk():
    radius = 0.6
    sdf_fn, color_fn = _sphere_scene(radius)
    h = w = 24
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), fov_deg=40, height=h, width=w)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=1.5, far=4.0, n_samples=128)
    opacity = out["opacity"].reshape(h, w)

    center = opacity[h // 2, w // 2]
    corner = opacity[0, 0]
    assert center > 0.9, f"sphere center should be opaque, got {center:.3f}"
    assert corner < 0.1, f"corner should see empty space, got {corner:.3f}"


def test_analytic_sphere_depth_matches_geometry():
    radius = 0.6
    dist = 3.0
    sdf_fn, color_fn = _sphere_scene(radius)
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -dist]), torch.zeros(3), fov_deg=40, height=24, width=24)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=1.5, far=4.0, n_samples=256)
    depth = out["depth"].reshape(24, 24)
    center_depth = float(depth[12, 12])
    # Front of the sphere sits at (dist - radius) along the central ray.
    assert abs(center_depth - (dist - radius)) < 0.1


def test_weights_are_a_valid_partition():
    sdf_fn, color_fn = _sphere_scene(0.6)
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -3.0]), torch.zeros(3), height=8, width=8)
    out = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, n_samples=64)
    w = out["weights"]
    assert torch.all(w >= 0)
    assert torch.all(out["opacity"] <= 1.0 + 1e-4)


def test_field_render_is_differentiable():
    torch.manual_seed(0)
    field = SpatialLatentField(TriPlaneConfig(cond_dim=16, plane_channels=8, grid_res=16, feature_dim=16))
    renderer = NeusRenderer(n_samples=32)
    planes = field.encode(torch.zeros(1, 16))
    o, d = look_at_rays(torch.tensor([0.0, 0.0, -2.0]), torch.zeros(3), height=6, width=6)
    out = renderer.render(field, planes, o, d, near=0.5, far=3.5)
    assert out["rgb"].shape == (36, 3)
    out["rgb"].mean().backward()
    assert any(p.grad is not None for p in field.parameters())
    assert renderer.deviation.variance.grad is not None


def test_photometric_overfit_to_sphere_views_decreases_loss():
    """End-to-end: supervise the field with posed images of a sphere."""
    torch.manual_seed(1)
    radius = 0.6
    sdf_fn, color_fn = _sphere_scene(radius)
    h = w = 14

    # Ground-truth renders from three viewpoints.
    cams = [
        torch.tensor([0.0, 0.0, -2.5]),
        torch.tensor([2.5, 0.0, 0.0]),
        torch.tensor([0.0, 2.2, -1.0]),
    ]
    views = []
    for c in cams:
        o, d = look_at_rays(c, torch.zeros(3), fov_deg=45, height=h, width=w)
        gt = volume_render_sdf(sdf_fn, color_fn, o, d, INV_S, near=0.5, far=4.0, n_samples=96)
        views.append(View(o, d, gt["rgb"], gt["opacity"].clamp(0, 1)))

    field = SpatialLatentField(
        TriPlaneConfig(cond_dim=16, plane_channels=16, grid_res=32, feature_dim=32, decoder_hidden=64)
    )
    renderer = NeusRenderer(n_samples=48)
    hist = fit_from_views(field, renderer, views, steps=120, rays_per_step=196, lr=3e-3, near=0.5, far=4.0)

    first = float(np.mean(hist[:15]))
    last = float(np.mean(hist[-15:]))
    assert last < 0.6 * first, f"photometric loss did not fall enough: {first:.4f} -> {last:.4f}"
