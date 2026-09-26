"""
DPM-Solver++ (Lu et al.) — data-prediction / x0 form for VP diffusion,
and Adams–Bashforth-style multistep for rectified-flow ODEs.

VP uses exponential integrators on λ = ½ log(ᾱ/(1-ᾱ)).
Flow uses multistep integration of dx/ds = v on s ∈ [1, 0].
"""

from __future__ import annotations

import math

import torch

from diffusion.solvers.base import SolverState


def lambda_from_alpha_cumprod(alpha_bar: torch.Tensor) -> torch.Tensor:
    """λ(t) = ½ log(ᾱ / (1-ᾱ))."""
    ab = alpha_bar.clamp(1e-12, 1.0 - 1e-12)
    return 0.5 * torch.log(ab / (1.0 - ab))


def _expand_coeff(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while c.ndim < x.ndim:
        c = c.unsqueeze(-1)
    return c


def vp_dpmpp_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    state: SolverState,
    order: int = 2,
) -> tuple[torch.Tensor, SolverState]:
    """
    One DPM-Solver++ step from current (noisier) → next (cleaner).

    ``alpha_bar_*`` are per-batch scalars shape (B,) or broadcastable.
    ``order`` is 1 (first-order), 2 (2M), or 3 (3M).
    """
    device = x.device
    dtype = x.dtype
    ab_c = alpha_bar_cur.to(device=device, dtype=torch.float64)
    ab_n = alpha_bar_next.to(device=device, dtype=torch.float64)
    if ab_c.ndim == 0:
        ab_c = ab_c.expand(x.shape[0])
        ab_n = ab_n.expand(x.shape[0])

    lam_c = lambda_from_alpha_cumprod(ab_c)
    lam_n = lambda_from_alpha_cumprod(ab_n)
    h = (lam_n - lam_c).clamp(min=1e-8)

    alpha_n = ab_n.sqrt()
    sigma_c = (1.0 - ab_c).clamp(min=1e-12).sqrt()
    sigma_n = (1.0 - ab_n).clamp(min=1e-12).sqrt()

    x0 = x0_pred.to(dtype=torch.float64)
    x64 = x.to(dtype=torch.float64)

    # Use effective order based on available history (after we push current).
    # Push current x0 at λ_cur before computing (matches diffusers multistep).
    state.push(x0.detach().to(dtype=dtype), float(lam_c.reshape(-1)[0].item()))

    n_hist = state.order
    use_order = min(int(order), n_hist)

    sigma_ratio = _expand_coeff(sigma_n / sigma_c.clamp(min=1e-12), x64)
    alpha_n_e = _expand_coeff(alpha_n, x64)
    exp_h = torch.exp(-h)
    phi_1 = _expand_coeff(exp_h - 1.0, x64)  # e^{-h}-1

    m0 = state.model_outputs[-1].to(dtype=torch.float64)

    if use_order == 1 or n_hist < 2:
        x_next = sigma_ratio * x64 - alpha_n_e * phi_1 * m0
    elif use_order == 2 or n_hist < 3:
        m1 = state.model_outputs[-2].to(dtype=torch.float64)
        lam_s1 = state.timesteps[-2]
        # h_0 = λ_cur - λ_prev (previous was noisier → smaller λ)
        h0 = float(lam_c.reshape(-1)[0].item()) - float(lam_s1)
        r0 = h0 / float(h.reshape(-1)[0].item() + 1e-12)
        r0 = max(r0, 1e-6)
        d0 = m0
        d1 = (1.0 / r0) * (m0 - m1)
        x_next = sigma_ratio * x64 - alpha_n_e * phi_1 * d0 - 0.5 * alpha_n_e * phi_1 * d1
    else:
        m1 = state.model_outputs[-2].to(dtype=torch.float64)
        m2 = state.model_outputs[-3].to(dtype=torch.float64)
        lam0 = float(lam_c.reshape(-1)[0].item())
        lam1 = float(state.timesteps[-2])
        lam2 = float(state.timesteps[-3])
        h0 = lam0 - lam1
        h1 = lam1 - lam2
        hh = float(h.reshape(-1)[0].item())
        r0 = max(h0 / (hh + 1e-12), 1e-6)
        r1 = max(h1 / (hh + 1e-12), 1e-6)
        d0 = m0
        d1_0 = (1.0 / r0) * (m0 - m1)
        d1_1 = (1.0 / r1) * (m1 - m2)
        # DPM-Solver++ 3M coefficients (Lu et al. / Diffusers)
        d1 = d1_0 + (1.0 / (r0 + r1 + 1e-12)) * (r0 * (d1_0 - d1_1))
        d2 = (1.0 / (r0 + r1 + 1e-12)) * (d1_0 - d1_1)
        x_next = (
            sigma_ratio * x64
            - alpha_n_e * phi_1 * d0
            - 0.5 * alpha_n_e * phi_1 * d1
            - (1.0 / 6.0) * alpha_n_e * (phi_1 + _expand_coeff(torch.exp(-h) + h - 1.0, x64)) * d2
        )

    return x_next.to(dtype=dtype), state


def flow_dpmpp_2m_update(
    *,
    x: torch.Tensor,
    velocity: torch.Tensor,
    s_cur: float,
    s_next: float,
    state: SolverState,
) -> tuple[torch.Tensor, SolverState]:
    """
    Multistep (Adams–Bashforth 2) update for rectified-flow ODE dx/ds = v.

    First step falls back to Euler; subsequent steps use 2M correction.
    """
    ds = float(s_next) - float(s_cur)
    state.push(velocity.detach(), float(s_cur))
    v0 = velocity
    if state.order < 2 or abs(ds) < 1e-12:
        x_next = x + v0 * ds
        return x_next, state

    v1 = state.model_outputs[-2]
    s_prev = state.timesteps[-2]
    ds_prev = float(s_cur) - float(s_prev)
    r = ds_prev / (ds + 1e-12) if abs(ds) > 1e-12 else 1.0
    r = max(abs(r), 1e-6) * math.copysign(1.0, r if r != 0 else 1.0)
    # Standard AB2 / DPM++-style 2M on constant-step ODE:
    # x += ds * ((1 + 1/(2r)) v0 - (1/(2r)) v_prev)
    c0 = 1.0 + 0.5 / r
    c1 = -0.5 / r
    x_next = x + ds * (c0 * v0 + c1 * v1)
    return x_next, state
