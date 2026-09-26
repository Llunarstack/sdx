"""
UniPC (Zhao et al.) — unified predictor-corrector for few-step VP sampling.

Implements a practical bh2-style UniPC on x0 predictions, compatible with
SDX's discrete VP schedule via λ(t) from ᾱ.
"""

from __future__ import annotations

import torch

from diffusion.solvers.base import SolverState
from diffusion.solvers.dpm_solver_pp import lambda_from_alpha_cumprod, vp_dpmpp_update


def _expand_coeff(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while c.ndim < x.ndim:
        c = c.unsqueeze(-1)
    return c


def vp_unipc_predict(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    state: SolverState,
) -> tuple[torch.Tensor, SolverState]:
    """UniPC predictor (DPM++ 2M)."""
    return vp_dpmpp_update(
        x=x,
        x0_pred=x0_pred,
        alpha_bar_cur=alpha_bar_cur,
        alpha_bar_next=alpha_bar_next,
        state=state,
        order=2,
    )


def vp_unipc_correct(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    corrector_x0: torch.Tensor,
    alpha_bar_cur: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    state: SolverState,
) -> tuple[torch.Tensor, SolverState]:
    """
    UniPC corrector: refine the step using x0 evaluated at the predicted next state.

    Does **not** push a new history entry (predictor already did); updates the
    last history slot to the corrected x0 at λ_next.
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

    x64 = x.to(dtype=torch.float64)
    m0 = x0_pred.to(dtype=torch.float64)
    m_corr = corrector_x0.to(dtype=torch.float64)
    m_avg = 0.5 * (m0 + m_corr)

    sigma_ratio = _expand_coeff(sigma_n / sigma_c.clamp(min=1e-12), x64)
    alpha_n_e = _expand_coeff(alpha_n, x64)
    phi_1 = _expand_coeff(torch.exp(-h) - 1.0, x64)

    if state.order >= 2:
        m1 = state.model_outputs[-2].to(dtype=torch.float64)
        lam_s1 = state.timesteps[-2]
        h0 = float(lam_c.reshape(-1)[0].item()) - float(lam_s1)
        r0 = max(h0 / float(h.reshape(-1)[0].item() + 1e-12), 1e-6)
        d0 = m_avg
        d1 = (1.0 / r0) * (m_avg - m1)
        x_next = sigma_ratio * x64 - alpha_n_e * phi_1 * d0 - 0.5 * alpha_n_e * phi_1 * d1
    else:
        x_next = sigma_ratio * x64 - alpha_n_e * phi_1 * m_avg

    if state.model_outputs:
        state.model_outputs[-1] = m_corr.detach().to(dtype=dtype)
        state.timesteps[-1] = float(lam_n.reshape(-1)[0].item())

    return x_next.to(dtype=dtype), state


def vp_unipc_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    state: SolverState,
    corrector_x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolverState, bool]:
    """
    UniPC step wrapper.

    - ``corrector_x0 is None``: run predictor, return ``(x_pred, state, True)``.
    - ``corrector_x0`` set: run corrector only, return ``(x_next, state, False)``.
    """
    if corrector_x0 is None:
        x_pred, state = vp_unipc_predict(
            x=x,
            x0_pred=x0_pred,
            alpha_bar_cur=alpha_bar_cur,
            alpha_bar_next=alpha_bar_next,
            state=state,
        )
        return x_pred, state, True
    x_next, state = vp_unipc_correct(
        x=x,
        x0_pred=x0_pred,
        corrector_x0=corrector_x0,
        alpha_bar_cur=alpha_bar_cur,
        alpha_bar_next=alpha_bar_next,
        state=state,
    )
    return x_next, state, False
