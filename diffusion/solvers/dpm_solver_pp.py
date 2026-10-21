"""DPM-Solver++ multistep data-prediction solvers (Lu et al. 2022, arXiv:2211.01095).

One exponential integrator for any interpolant ``x = alpha*x0 + sigma*eps`` on
``lambda = log(alpha/sigma)``: VP uses ``alpha = sqrt(abar)``, ``sigma = sqrt(1-abar)``;
rectified flow uses ``alpha = 1-s``, ``sigma = s`` with ``x0 = x - s*v``. ``eta > 0`` gives the
DPM-Solver++(2M) SDE update (noise ``sigma_next*sqrt(1-exp(-2*eta*h))*z``); ``eta = 0`` is the ODE.
"""

from __future__ import annotations

import math

import torch

from diffusion.solvers.base import SolverState


def _lam(alpha: float, sigma: float) -> float:
    return math.log(max(alpha, 1e-12)) - math.log(max(sigma, 1e-12))


def dpmpp_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_cur: float,
    sigma_cur: float,
    alpha_next: float,
    sigma_next: float,
    state: SolverState,
    order: int = 2,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolverState]:
    """One DPM-Solver++ step (orders 1-3; ``eta`` > 0 only up to order 2) from the current to the next level.

    A step onto ``sigma_next == 0`` returns the data prediction: lambda is infinite there, and the
    first-order update is exact (diffusers' ``final_sigmas_type="zero"``).
    """
    lam_c = _lam(alpha_cur, sigma_cur)
    state.push(x0_pred.detach(), lam_c)
    if sigma_next <= 0.0:
        return x0_pred, state
    use = min(int(order), state.order)
    if eta > 0.0 and use > 2:
        raise ValueError("the DPM-Solver++ SDE update is implemented up to order 2")
    h = max(_lam(alpha_next, sigma_next) - lam_c, 1e-8)
    eh = float(eta) * h
    phi = -math.expm1(-h - eh)
    d0 = x0_pred
    x_next = (sigma_next / max(sigma_cur, 1e-12)) * math.exp(-eh) * x + alpha_next * phi * d0
    if use >= 2:
        r0 = max((lam_c - state.timesteps[-2]) / h, 1e-6)
        d1_0 = (d0 - state.model_outputs[-2]) / r0
        if use == 2:
            x_next = x_next + 0.5 * alpha_next * phi * d1_0
        else:
            r1 = max((state.timesteps[-2] - state.timesteps[-3]) / h, 1e-6)
            d1_1 = (state.model_outputs[-2] - state.model_outputs[-3]) / r1
            d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            x_next = x_next + alpha_next * (1.0 - phi / h) * d1 - alpha_next * ((h - phi) / h**2 - 0.5) * d2
    if eh > 0.0:
        z = torch.randn_like(x) if noise is None else noise
        x_next = x_next + sigma_next * math.sqrt(-math.expm1(-2.0 * eh)) * z
    return x_next, state


def _vp_alpha_sigma(alpha_bar) -> tuple[float, float]:
    """``(sqrt(abar), sqrt(1 - abar))`` from the first element of ``alpha_bar`` (uniform over the batch)."""
    ab = float(torch.as_tensor(alpha_bar).reshape(-1)[0])
    return math.sqrt(ab), math.sqrt(max(1.0 - ab, 0.0))


def vp_dpmpp_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor | float,
    alpha_bar_next: torch.Tensor | float,
    state: SolverState,
    order: int = 2,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolverState]:
    """DPM-Solver++ on the VP schedule; ``order`` 1, 2 (2M) or 3 (3M). ᾱ is uniform over the batch."""
    alpha_cur, sigma_cur = _vp_alpha_sigma(alpha_bar_cur)
    alpha_next, sigma_next = _vp_alpha_sigma(alpha_bar_next)
    return dpmpp_update(
        x=x,
        x0_pred=x0_pred,
        alpha_cur=alpha_cur,
        sigma_cur=sigma_cur,
        alpha_next=alpha_next,
        sigma_next=sigma_next,
        state=state,
        order=order,
        eta=eta,
        noise=noise,
    )


def flow_dpmpp_2m_update(
    *,
    x: torch.Tensor,
    velocity: torch.Tensor,
    s_cur: float,
    s_next: float,
    state: SolverState,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolverState]:
    """DPM-Solver++(2M) on rectified flow ``x = (1-s)*x0 + s*eps``; the first step reduces to Euler."""
    s_c, s_n = float(s_cur), float(s_next)
    return dpmpp_update(
        x=x,
        x0_pred=x - s_c * velocity,
        alpha_cur=1.0 - s_c,
        sigma_cur=s_c,
        alpha_next=1.0 - s_n,
        sigma_next=s_n,
        state=state,
        order=2,
        eta=eta,
        noise=noise,
    )
