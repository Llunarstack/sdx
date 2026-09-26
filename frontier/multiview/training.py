"""
Training objective for the spatial latent field.

An untrained field meshes to noise because nothing has told it what "inside" and
"outside" mean. This module supplies that signal. Two losses do the real work:

  * **SDF regression** — direct supervision toward a target signed distance
    (from an analytic shape, or the precomputed SDF of a reference mesh / scan).
    This is what pins the geometry.
  * **Eikonal regularization** — the gradient of a true signed distance field has
    unit norm everywhere (``||∇sdf|| = 1``). Penalizing departures from that turns
    the network's arbitrary level set into a *metric* distance field, which is
    exactly the property marching tetrahedra relies on to produce clean, watertight
    geometry. Without it you get blobby, non-manifold surfaces.

``fit_sdf`` ties them together and can overfit the field to a single target — the
verifiable base case (train on a sphere, mesh it, get a sphere). Multi-view
*photometric* supervision (render the field, compare to ground-truth images) is
the next layer and plugs into the same optimizer loop via ``render_view``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import torch
import torch.nn.functional as F
from torch import Tensor

from .triplane import SpatialLatentField

SdfTarget = Callable[[Tensor], Tensor]  # points (B, N, 3) -> sdf (B, N, 1)


def sample_points(batch: int, n: int, *, device: torch.device | str = "cpu") -> Tensor:
    """Uniform points in the ``[-1, 1]^3`` field cube, shape ``(batch, n, 3)``."""
    return torch.empty(batch, n, 3, device=device).uniform_(-1.0, 1.0)


def sdf_and_gradient(
    field: SpatialLatentField, planes: tuple[Tensor, Tensor, Tensor], points: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(sdf, d sdf / d points)``. The gradient is what eikonal acts on.

    ``create_graph=True`` keeps the gradient differentiable so the eikonal penalty
    itself trains the network (a second-order term — this is why the tri-plane uses
    a hand-written bilinear sampler that supports double backward).
    """
    points = points.detach().requires_grad_(True)
    sdf = field.decode_sdf(planes, points)
    grad = torch.autograd.grad(sdf, points, grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
    return sdf, grad


def eikonal_loss(grad: Tensor) -> Tensor:
    """Penalize deviation of the SDF gradient norm from 1."""
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def sdf_regression_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE toward a target SDF, with extra weight near the surface.

    Errors close to the zero level set matter far more than errors deep inside or
    far outside, so weight samples by how near the target places them.
    """
    weight = torch.exp(-4.0 * target.abs())
    return (weight * (pred - target) ** 2).mean()


def surface_normal(field: SpatialLatentField, planes: tuple[Tensor, Tensor, Tensor], points: Tensor) -> Tensor:
    """Unit surface normal = normalized SDF gradient (for shading / normal loss)."""
    _, grad = sdf_and_gradient(field, planes, points)
    return F.normalize(grad, dim=-1)


@dataclass(slots=True)
class FitResult:
    losses: list[float] = dataclass_field(default_factory=list)
    eikonal: list[float] = dataclass_field(default_factory=list)
    cond: Tensor | None = None


def fit_sdf(
    field: SpatialLatentField,
    target_sdf: SdfTarget,
    *,
    cond: Tensor | None = None,
    steps: int = 300,
    lr: float = 1e-3,
    n_points: int = 2048,
    eikonal_weight: float = 0.1,
    device: torch.device | str = "cpu",
) -> FitResult:
    """Overfit ``field`` to a single target SDF shape.

    ``cond`` is the fixed conditioning vector the shape is stored under (defaults
    to zeros); training adjusts the field weights so ``encode(cond)`` reproduces
    the target. Returns the loss history and the ``cond`` used (so the caller can
    ``encode`` + ``to_mesh`` afterwards).
    """
    field.to(device).train()
    if cond is None:
        cond = torch.zeros(1, field.cfg.cond_dim, device=device)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    result = FitResult(cond=cond)

    for _ in range(steps):
        planes = field.encode(cond)
        points = sample_points(cond.shape[0], n_points, device=device)
        target = target_sdf(points)

        pred, grad = sdf_and_gradient(field, planes, points)
        reg = sdf_regression_loss(pred, target)
        eik = eikonal_loss(grad)
        loss = reg + eikonal_weight * eik

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        result.losses.append(float(reg.detach()))
        result.eikonal.append(float(eik.detach()))

    field.eval()
    return result


def sphere_sdf(radius: float) -> SdfTarget:
    """Convenience analytic target: a centered sphere of the given radius."""

    def fn(points: Tensor) -> Tensor:
        return points.norm(dim=-1, keepdim=True) - radius

    return fn
