"""
Minibatch optimal-transport coupling between data latents and Gaussian noise (OT-CFM: Tong et al. 2023,
"Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"; Pooladian et al. 2023).

Cost ``C_ij = ||x0_i - eps_j||^2``. The noise rows are only **permuted**, never mixed, so the batch still holds the
same i.i.d. N(0, I) draws; pairing each latent with a nearby draw straightens the paths the model has to learn.

- ``hungarian``: exact OT permutation (scipy ``linear_sum_assignment``).
- ``soft``: entropic OT plan from log-domain Sinkhorn (``reg`` in units of the batch cost std), then a permutation
  sampled from it row by row without replacement (``reg ~ 0.05`` is close to the exact assignment, ``reg >~ 1`` near
  random).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

PairMode = Literal["soft", "hungarian"]


def sinkhorn_log_plan(cost: torch.Tensor, *, reg: float, n_iters: int) -> torch.Tensor:
    """Log of the entropic OT plan with uniform marginals (rows and columns each sum to ``1/B``).

    Log-domain updates (Peyre & Cuturi 2019, Sec. 4.4) so ``cost / reg`` in the thousands does not underflow.
    """
    b = cost.shape[0]
    log_k = -cost.double() / max(float(reg), 1e-12)
    log_marginal = torch.full((b,), -math.log(b), dtype=torch.float64, device=cost.device)
    f = torch.zeros_like(log_marginal)
    g = torch.zeros_like(log_marginal)
    for _ in range(max(1, int(n_iters))):
        f = log_marginal - torch.logsumexp(log_k + g[None, :], dim=1)
        g = log_marginal - torch.logsumexp(log_k + f[:, None], dim=0)
    return log_k + f[:, None] + g[None, :]


def sample_permutation_from_plan(log_plan: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Column index per row, each column used once: rows in random order draw from their plan row over free columns."""
    b = log_plan.shape[0]
    free = torch.ones(b, dtype=torch.bool, device=log_plan.device)
    perm = torch.empty(b, dtype=torch.long, device=log_plan.device)
    for i in torch.randperm(b, generator=generator).tolist():
        logits = log_plan[i].masked_fill(~free, float("-inf"))
        j = int(torch.multinomial(torch.softmax(logits, dim=0).float().cpu(), 1, generator=generator))
        perm[i] = j
        free[j] = False
    return perm


def pair_noise_to_latents(
    latents: torch.Tensor,
    noise: torch.Tensor,
    *,
    reg: float = 0.05,
    n_iters: int = 40,
    mode: PairMode = "soft",
) -> torch.Tensor:
    """Return ``noise`` with its batch rows permuted so row ``i`` is paired with ``latents[i]`` by (entropic) OT."""
    if latents.shape != noise.shape:
        raise ValueError("latents and noise must have the same shape")
    b = latents.shape[0]
    if b < 2 or reg <= 0.0:
        return noise
    cost = torch.cdist(latents.reshape(b, -1).float(), noise.reshape(b, -1).float()) ** 2
    if mode == "soft":
        # ``reg`` is in units of the cost spread (std), so its meaning does not depend on latent size.
        cost = (cost - cost.mean()) / cost.std().clamp(min=1e-12)
        perm = sample_permutation_from_plan(sinkhorn_log_plan(cost, reg=reg, n_iters=n_iters))
        return noise[perm]
    if mode == "hungarian":
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = torch.empty(b, dtype=torch.long, device=noise.device)
        perm[torch.as_tensor(rows, device=noise.device)] = torch.as_tensor(cols, device=noise.device)
        return noise[perm]
    raise ValueError(f"Unknown mode {mode!r}")


__all__ = ["pair_noise_to_latents", "sample_permutation_from_plan", "sinkhorn_log_plan"]
