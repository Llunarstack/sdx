"""
Linear **rectified-flow-style** training path (prototype).

Uses ``x_t = (1 - s) x_0 + s \\epsilon`` with ``s = t / (T-1)`` and discrete ``t \\in [0, T-1]`` for
the DiT timestep embedding. Target **velocity** is ``v^* = \\epsilon - x_0``; the network output is
matched to ``v^*`` (same channels as latents).

Sampling: use ``GaussianDiffusion.sample_loop(..., flow_matching_sample=True)`` (Euler or Heun in
``s`` with the same ``t`` index map as training). VP ``sample_loop`` remains the default when this
flag is off.

See ``docs/MODERN_DIFFUSION.md`` and ``docs/CONSISTENCY_FLOW_SPEED_BLUEPRINT.md``.
"""

from __future__ import annotations

import torch


def flow_matching_per_sample_losses(
    model: torch.nn.Module,
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    num_timesteps: int,
    model_kwargs: dict,
) -> torch.Tensor:
    """
    Per-row MSE matching ``model(x_t, t)`` to ``epsilon - x0``.

    Returns tensor of shape ``(B,)``.
    """
    if x0.shape != epsilon.shape:
        raise ValueError("x0 and epsilon must match shape")
    b, _, _, _ = x0.shape
    device = x0.device
    t_idx = torch.randint(0, int(num_timesteps), (b,), device=device, dtype=torch.long)
    denom = max(int(num_timesteps) - 1, 1)
    st = (t_idx.float() / float(denom)).view(b, 1, 1, 1)
    x_t = (1.0 - st) * x0 + st * epsilon
    v_star = epsilon - x0
    out = model(x_t, t_idx, **model_kwargs)
    if out.shape != x0.shape and out.shape[1] > x0.shape[1]:
        out = out[:, : x0.shape[1]]
    return (out - v_star).pow(2).mean(dim=(1, 2, 3))


__all__ = ["flow_matching_per_sample_losses"]
