"""UniPC (Zhao et al., NeurIPS 2023, arXiv:2302.04867), data prediction with ``B(h) = e^{-h} - 1`` (bh2).

The predictor UniP-2 equals the DPM-Solver++(2M) step (``vp_dpmpp_update``). The corrector UniC
re-solves the last step with the model output at the predicted point, which the next step
evaluates anyway, so UniPC costs one forward per step. UniC-2 reduces to Adams-Moulton
``(5/12, 8/12, -1/12)`` weights on a uniform λ grid.
"""

from __future__ import annotations

import math

import torch

from diffusion.solvers.base import SolverState
from diffusion.solvers.dpm_solver_pp import _lam, _vp_alpha_sigma


def unipc_correct(
    *,
    x_prev: torch.Tensor,
    x0_new: torch.Tensor,
    alpha_prev: float,
    sigma_prev: float,
    alpha_cur: float,
    sigma_cur: float,
    state: SolverState,
    order: int = 2,
) -> torch.Tensor:
    """UniC: corrected sample at the current level from ``x_prev`` (the predictor's input) and ``x0_new``.

    ``state`` must still end with the data prediction at the previous level (call before pushing ``x0_new``).
    """
    m0, lam0 = state.model_outputs[-1], state.timesteps[-1]
    h = max(_lam(alpha_cur, sigma_cur) - lam0, 1e-8)
    phi = -math.expm1(-h)
    x_next = (sigma_cur / max(sigma_prev, 1e-12)) * x_prev + alpha_cur * phi * m0
    d_t = x0_new - m0
    if order < 2 or state.order < 2:
        return x_next + 0.5 * alpha_cur * phi * d_t
    r1 = min((state.timesteps[-2] - lam0) / h, -1e-6)
    d1 = (state.model_outputs[-2] - m0) / r1
    b_h = math.expm1(-h)
    k1 = b_h / -h - 1.0
    b0 = k1 / b_h
    b1 = 2.0 * (k1 / -h - 0.5) / b_h
    rho0 = (b0 - b1) / (1.0 - r1)
    rho1 = b0 - rho0
    return x_next + alpha_cur * phi * (rho0 * d1 + rho1 * d_t)


def vp_unipc_correct(
    *,
    x_prev: torch.Tensor,
    x0_new: torch.Tensor,
    alpha_bar_prev,
    alpha_bar_cur,
    state: SolverState,
    order: int = 2,
) -> torch.Tensor:
    """``unipc_correct`` on the VP schedule (ᾱ uniform over the batch)."""
    alpha_prev, sigma_prev = _vp_alpha_sigma(alpha_bar_prev)
    alpha_cur, sigma_cur = _vp_alpha_sigma(alpha_bar_cur)
    return unipc_correct(
        x_prev=x_prev,
        x0_new=x0_new,
        alpha_prev=alpha_prev,
        sigma_prev=sigma_prev,
        alpha_cur=alpha_cur,
        sigma_cur=sigma_cur,
        state=state,
        order=order,
    )
