"""
**Structured noise** for DiT latents: diversity / shallow ensembles without a second model.

These are cheap alternatives to full particle filters when exploring test-time scaling.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def orthogonal_noise_directions(
    shape: Tuple[int, ...],
    k: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return ``k`` flattened noise vectors of length ``prod(shape)``, orthonormalized (QR).

    Use as diverse perturbation axes: ``x + sum_i eps_i * u_i``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    flat = int(torch.tensor(shape).prod().item())
    a = torch.randn(flat, k, generator=generator, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.T.reshape(k, *shape)


def low_rank_latent_jitter(
    x: torch.Tensor,
    rank: int,
    scale: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Add ``scale * (u @ v)``-style noise with learnable-free random ``u,v`` along last dim.

    ``x`` shape ``(..., C)``; rank ``r`` uses ``r`` degrees of freedom per spatial position.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    *lead, c = x.shape
    if c < rank:
        raise ValueError("last dim must be >= rank")
    u = torch.randn(*lead, rank, generator=generator, device=x.device, dtype=x.dtype)
    v = torch.randn(rank, c, generator=generator, device=x.device, dtype=x.dtype)
    delta = torch.einsum("...r,rc->...c", u, v)
    return x + float(scale) * delta


def weighted_ensemble_latents(
    samples: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    dim: int = 0,
) -> torch.Tensor:
    """Stack ``samples`` along ``dim`` and take a weighted average (broadcast-safe)."""
    if not samples:
        raise ValueError("samples must be non-empty")
    if len(samples) != len(weights):
        raise ValueError("samples and weights length mismatch")
    w = torch.tensor(weights, dtype=samples[0].dtype, device=samples[0].device)
    w = w / w.sum().clamp_min(1e-8)
    stacked = torch.stack(list(samples), dim=dim)
    # move weights to broadcast: [..., 1, 1, ...] on ensemble dim
    view_shape = [1] * stacked.dim()
    view_shape[dim] = -1
    ww = w.reshape(*view_shape)
    return (stacked * ww).sum(dim=dim)
