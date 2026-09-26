"""
Genetic cross-breeding of 3D objects.

The wildest idea from the original SDX-3D notes: treat every asset as a genotype
and *breed* two together — "Victorian armchair × desert scorpion → chitin chair,"
where the tail becomes the spine and the chitin plates become upholstery. Two
handles here:

  * **Latent blend** (``blend_conditioning``) — spherically interpolate the two
    conditioning vectors and decode one coherent hybrid. This is the "genetic
    average": features fuse semantically rather than being glued together.
  * **Spatial graft** (``spatial_graft``) — take the lower part of object A and the
    upper part of object B (or any axis), blended smoothly across the seam. This
    is the literal chimera: A's legs, B's torso.

Both return things the rest of the stack already understands (a cond vector, or an
SDF callable you can mesh / validate).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

NumpySdf = Callable[[np.ndarray], np.ndarray]


def lerp(a: Tensor, b: Tensor, t: float) -> Tensor:
    """Straight-line interpolation between two conditioning vectors."""
    return (1.0 - t) * a + t * b


def slerp(a: Tensor, b: Tensor, t: float, *, eps: float = 1e-7) -> Tensor:
    """Spherical interpolation — travels along the hypersphere, preserving norm.

    Better than ``lerp`` for latent codes: it keeps the interpolant on the shell
    the two parents live on, so the midpoint is a plausible object rather than a
    washed-out average that decodes to mush.

    Near-parallel inputs (|sin ω| tiny) fall back to lerp to avoid NaN from the
    division by sin(ω). Unit vectors are interpolated then rescaled to the
    lerped parent norms.
    """
    a_flat, b_flat = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    a_norm = a_flat.norm(dim=-1, keepdim=True)
    b_norm = b_flat.norm(dim=-1, keepdim=True)
    na = a_flat / (a_norm + eps)
    nb = b_flat / (b_norm + eps)
    dot = (na * nb).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Fall back to lerp when vectors are nearly collinear (ω ≈ 0 or π).
    use_lerp = sin_omega.abs() < 1e-4
    wa = torch.sin((1.0 - t) * omega) / sin_omega.clamp(min=eps)
    wb = torch.sin(t * omega) / sin_omega.clamp(min=eps)
    unit = wa * na + wb * nb
    unit_lerp = (1.0 - t) * na + t * nb
    unit = torch.where(use_lerp, unit_lerp, unit)
    out_norm = (1.0 - t) * a_norm + t * b_norm
    out = unit * out_norm
    return out.reshape_as(a)


def blend_conditioning(cond_a: Tensor, cond_b: Tensor, t: float = 0.5) -> Tensor:
    """Breed two objects at the latent level (semantic fusion). ``t`` biases toward B."""
    return slerp(cond_a, cond_b, t)


def spatial_graft(
    sdf_a: NumpySdf,
    sdf_b: NumpySdf,
    *,
    axis: int = 1,
    split: float = 0.0,
    smooth: float = 0.08,
) -> NumpySdf:
    """Chimera: object A below ``split`` on ``axis``, object B above, blended.

    The seam uses a logistic weight so the two bodies fuse continuously instead of
    showing a hard cut, which keeps the grafted surface watertight and meshable.
    """

    def grafted(points: np.ndarray) -> np.ndarray:
        coord = points[:, axis]
        w = 1.0 / (1.0 + np.exp(-(coord - split) / smooth))  # 0 below seam -> 1 above
        return (1.0 - w) * sdf_a(points) + w * sdf_b(points)

    return grafted


def mutate_conditioning(cond: Tensor, *, strength: float = 0.1, seed: int | None = None) -> Tensor:
    """Random mutation of a genotype — explore nearby variants of one object."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=cond.device).manual_seed(seed)
    noise = torch.randn(cond.shape, generator=gen, device=cond.device, dtype=cond.dtype)
    return cond + strength * noise
