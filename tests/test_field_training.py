"""Tests for the spatial-field training objective.

The headline test overfits the field to a sphere and confirms it then *meshes*
to a sphere — proof the objective teaches real geometry, not just that a loss
number goes down.
"""

from __future__ import annotations

import numpy as np
import torch
from frontier.multiview import (
    SpatialLatentField,
    TriPlaneConfig,
    eikonal_loss,
    fit_sdf,
    sdf_and_gradient,
    sdf_regression_loss,
    sphere_sdf,
    surface_normal,
)


def _small_field(seed: int = 0) -> SpatialLatentField:
    torch.manual_seed(seed)
    return SpatialLatentField(
        TriPlaneConfig(cond_dim=32, plane_channels=16, grid_res=32, feature_dim=32, decoder_hidden=64)
    )


def test_eikonal_zero_for_unit_gradients() -> None:
    grad = torch.nn.functional.normalize(torch.randn(2, 50, 3), dim=-1)
    assert eikonal_loss(grad).item() < 1e-6


def test_eikonal_positive_for_nonunit_gradients() -> None:
    grad = 2.0 * torch.nn.functional.normalize(torch.randn(2, 50, 3), dim=-1)
    assert abs(eikonal_loss(grad).item() - 1.0) < 1e-5  # (2 - 1)^2 = 1


def test_regression_weights_surface_more_than_interior() -> None:
    # Same absolute error (0.2), once near the surface (target≈0) and once far
    # (target=1). The near-surface error must be penalized more.
    near = sdf_regression_loss(torch.tensor([[[0.2]]]), torch.tensor([[[0.0]]]))
    far = sdf_regression_loss(torch.tensor([[[1.2]]]), torch.tensor([[[1.0]]]))
    assert near > far


def test_sdf_and_gradient_supports_double_backward() -> None:
    """Eikonal needs a differentiable gradient; verify second-order flow works
    through grid_sample and reaches the network parameters."""
    field = _small_field()
    planes = field.encode(torch.zeros(1, 32))
    points = torch.empty(1, 64, 3).uniform_(-1, 1)
    sdf, grad = sdf_and_gradient(field, planes, points)
    assert sdf.shape == (1, 64, 1)
    assert grad.shape == (1, 64, 3)
    eikonal_loss(grad).backward()
    assert any(p.grad is not None for p in field.parameters())


def test_surface_normal_is_unit() -> None:
    field = _small_field()
    planes = field.encode(torch.zeros(1, 32))
    n = surface_normal(field, planes, torch.empty(1, 40, 3).uniform_(-1, 1))
    norms = n.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_fit_sphere_learns_and_meshes_to_a_sphere() -> None:
    field = _small_field(seed=3)
    radius = 0.5
    result = fit_sdf(field, sphere_sdf(radius), steps=250, n_points=1024, lr=2e-3)

    # 1) The regression loss dropped substantially.
    first = float(np.mean(result.losses[:20]))
    last = float(np.mean(result.losses[-20:]))
    assert last < 0.3 * first, f"loss did not fall enough: {first:.4f} -> {last:.4f}"

    # 2) The trained field now meshes to a real, non-empty surface...
    planes = field.encode(result.cond)
    verts, faces = field.to_mesh(planes, resolution=32)
    assert len(verts) > 0 and len(faces) > 0

    # 3) ...and that surface is a sphere of about the right radius.
    r = np.linalg.norm(verts, axis=1)
    assert 0.35 < float(np.median(r)) < 0.65
