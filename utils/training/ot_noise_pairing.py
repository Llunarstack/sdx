"""
Mini-batch **optimal-transport-style** coupling between Gaussian noise and data latents.

Uses cost ``C_ij = ||vec(x0_i) - vec(eps_j)||^2`` and either **Sinkhorn** (soft ``P @ eps``)
or **Hungarian** (hard permutation). Intended as an experimental analogue of OT-coupled
rectified-flow training (see ``docs/BLUEPRINTS.md``).

This does **not** implement full rectified-flow objectives; it only reshapes which noise
vectors are paired with which latents inside a VP-DDPM training step.
"""

from __future__ import annotations

from typing import Literal

import torch

PairMode = Literal["soft", "hungarian"]


def sinkhorn_plan(cost: torch.Tensor, *, reg: float, n_iters: int) -> torch.Tensor:
    """
    Balanced optimal transport plan P (B, B) from Sinkhorn scaling.
    ``cost`` (B, B) non-negative; rows/columns sum to approximately 1/B each (uniform marginals).
    """
    B = cost.shape[0]
    if B < 2:
        return torch.eye(B, device=cost.device, dtype=cost.dtype)
    r = max(float(reg), 1e-8)
    K = torch.exp(-cost / r)
    a = torch.ones(B, device=cost.device, dtype=cost.dtype) / B
    b = torch.ones(B, device=cost.device, dtype=cost.dtype) / B
    u = torch.ones(B, device=cost.device, dtype=cost.dtype) / B
    v = torch.ones(B, device=cost.device, dtype=cost.dtype) / B
    for _ in range(max(1, int(n_iters))):
        u = a / (K @ v).clamp(min=1e-12)
        v = b / (K.T @ u).clamp(min=1e-12)
    return u[:, None] * K * v[None, :]


def pair_noise_to_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    reg: float = 0.05,
    n_iters: int = 40,
    mode: PairMode = "soft",
) -> torch.Tensor:
    """
    Return noise tensor same shape as ``noise``, coupled to ``latents`` along batch dim.

    - ``soft``: ``P @ noise_flat`` with Sinkhorn P from cost between latents and noise.
    - ``hungarian``: permute noise rows to minimize sum of paired costs (needs scipy).
    """
    if latents.shape != noise.shape:
        raise ValueError("latents and noise must have the same shape")
    B = latents.shape[0]
    if B < 2 or reg <= 0.0:
        return noise
    lf = latents.reshape(B, -1).to(dtype=torch.float32)
    nf = noise.reshape(B, -1).to(dtype=torch.float32)
    C = torch.cdist(lf, nf, p=2) ** 2
    if mode == "soft":
        P = sinkhorn_plan(C, reg=reg, n_iters=n_iters)
        mixed = (P @ nf).to(dtype=noise.dtype)
        return mixed.view_as(noise)
    if mode == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return noise
        c = C.detach().float().cpu().numpy()
        r_ind, c_ind = linear_sum_assignment(c)
        ri = torch.as_tensor(r_ind, device=noise.device, dtype=torch.long)
        ci = torch.as_tensor(c_ind, device=noise.device, dtype=torch.long)
        out = torch.zeros_like(noise)
        out[ri] = noise[ci]
        return out
    raise ValueError(f"Unknown mode {mode!r}")


__all__ = ["pair_noise_to_latents", "sinkhorn_plan"]
