"""APEX (Adaptive Predictive EXtrapolation): sampler + schedule for VP diffusion and rectified flow.

- Static schedule: a three-phase step-size profile over noise progress (structure, reconstruction,
  refinement) on top of a Karras-like power warp; ``phase_weight`` is the relative step size.
- Embedded error estimate at no extra NFE: DPM++ order 1 vs the phase order from the same ``x0``
  prediction (DPM-Solver-12 style).
- Difficulty from the error, ‖Δx0‖ velocity/curvature and an optional top-k spatial score; the
  adaptive VP variant re-spaces the remaining timesteps by it while keeping the NFE budget.
- Phase-aware order: structure ≤2, reconstruction up to ``max_order``, refinement ≤2.

References: Lu et al. DPM-Solver / DPM-Solver++ adaptive step size; Karras EDM σ schedules;
Align-Your-Steps knot densities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from diffusion.solvers.base import SolverState
from diffusion.solvers.dpm_solver_pp import flow_dpmpp_2m_update, vp_dpmpp_update

__all__ = [
    "ApexController",
    "ApexStepResult",
    "build_apex_vp_indices",
    "build_apex_flow_s_grid",
    "phase_weight",
    "vp_apex_update",
    "flow_apex_update",
    "redistribute_remaining_vp_indices",
]


def phase_weight(progress: float) -> float:
    """Relative step size at ``progress`` in [0, 1] (noisy → clean); peaks in the mid band."""
    p = max(0.0, min(1.0, float(progress)))
    mid = math.exp(-((p - 0.42) ** 2) / (2 * 0.18**2))
    late = 0.35 * math.exp(-((p - 0.82) ** 2) / (2 * 0.12**2))
    early = 0.45 + 0.25 * (1.0 - p)
    return float(early + 1.35 * mid + late)


def _apex_density_curve(n: int, *, rho: float = 7.0) -> np.ndarray:
    """Normalized step sizes for ``n`` intervals (sums to 1)."""
    n = max(1, int(n))
    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    karras = u ** (1.0 / float(max(rho, 1.0)))
    dens = np.array([phase_weight(float(ui)) for ui in u], dtype=np.float64) * (0.65 + 0.35 * karras)
    dens = np.maximum(dens, 1e-8)
    return dens / dens.sum()


def _enforce_desc(idx: np.ndarray, num_train: int) -> np.ndarray:
    """Coerce knots to a *strictly* descending integer schedule ending at 0.

    λ(t) saturates near the clean end, so many density targets round onto the
    same low indices (0/1/2). Forcing descent with a single forward pass turns
    that pile-up into a run of zeros — repeated t=0 steps that burn NFE on a
    zero-length interval. Reserving room from the tail first (``idx[j] >=
    n-1-j``) keeps every step on a distinct timestep, which is feasible
    whenever ``n <= num_train``.
    """
    arr = np.clip(np.asarray(idx, dtype=np.int64).reshape(-1), 0, max(int(num_train) - 1, 0))
    n = int(arr.size)
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    if n > int(num_train):
        # Distinct knots are impossible: fall back to an even spread rather than a zero run.
        return np.clip(np.round(np.linspace(num_train - 1, 0, n)).astype(np.int64), 0, num_train - 1)
    out = np.maximum(arr, np.arange(n - 1, -1, -1, dtype=np.int64))  # step j needs n-1-j knots below it
    for j in range(1, n):
        if out[j] >= out[j - 1]:
            out[j] = out[j - 1] - 1  # stays >= n-1-j because out[j-1] >= n-j
    out[-1] = 0
    return out


def build_apex_vp_indices(
    num_train: int,
    num_infer: int,
    alpha_cumprod: np.ndarray,
    *,
    rho: float = 7.0,
) -> np.ndarray:
    """Static APEX VP schedule: ``max(2, num_infer)`` descending integer indices (noisy → clean)."""
    T = int(num_train)
    N = max(2, int(num_infer))
    ac = np.clip(np.asarray(alpha_cumprod, dtype=np.float64).reshape(-1)[:T], 1e-12, 1.0 - 1e-12)
    # λ(t) = ½ log(ᾱ/(1-ᾱ)); t=0 clean (high λ), t=T-1 noisy (low λ)
    lam = 0.5 * np.log(ac / (1.0 - ac))
    lam_noisy, lam_clean = float(lam[-1]), float(lam[0])
    if abs(lam_clean - lam_noisy) < 1e-12:
        return np.linspace(T - 1, 0, N, dtype=np.int64)
    cum = np.concatenate([[0.0], np.cumsum(_apex_density_curve(N - 1, rho=rho))])
    lam_targets = lam_noisy + cum * (lam_clean - lam_noisy)
    idx = np.abs(lam[None, :] - lam_targets[:, None]).argmin(axis=1)
    return _enforce_desc(idx, T)


def build_apex_flow_s_grid(
    num_steps: int,
    *,
    s_start: float = 1.0,
    s_end: float = 0.0,
    rho: float = 7.0,
) -> np.ndarray:
    """Continuous flow ``s`` grid (``num_steps + 1`` knots) with the APEX three-phase density."""
    cum = np.concatenate([[0.0], np.cumsum(_apex_density_curve(num_steps, rho=rho))])
    grid = float(s_start) + (float(s_end) - float(s_start)) * cum
    # cumsum lands on 1±eps, which can push the final knot just past ``s_end``. Pin both ends.
    grid[0] = float(s_start)
    grid[-1] = float(s_end)
    return grid


def redistribute_remaining_vp_indices(
    t_cur: int,
    remaining_steps: int,
    num_train: int,
    alpha_cumprod: np.ndarray,
    *,
    difficulty: float = 1.0,
    rho: float = 7.0,
) -> np.ndarray:
    """Rebuild the ``remaining_steps`` indices strictly below ``t_cur`` down to 0.

    ``difficulty`` > 1 enlarges the steps right after ``t_cur`` and shrinks the late ones (< 1 the reverse).
    ``alpha_cumprod`` is unused: the tail is spaced in index space.
    """
    T = int(num_train)
    t0 = int(np.clip(t_cur, 1, T - 1))
    # Every knot must be distinct and strictly below ``t_cur``, so only ``t0`` of them can fit.
    # Returning the shorter feasible tail lets the caller keep its existing schedule (it checks
    # the length) instead of getting duplicate zeros or knots above the current time.
    rem = min(max(1, int(remaining_steps)), t0)
    if rem <= 1:
        return np.asarray([0], dtype=np.int64)
    d = float(max(difficulty, 0.25))
    dens = _apex_density_curve(rem - 1, rho=rho) * np.linspace(d, 1.0 / max(d, 1e-3), rem - 1)
    cum = np.concatenate([[0.0], np.cumsum(dens / dens.sum())])
    idx = np.round(t0 - cum * t0).astype(np.int64)
    idx[0] = min(int(idx[0]), t0 - 1)
    return _enforce_desc(idx, T)


@dataclass
class ApexController:
    """Error/difficulty tracker shared across the steps of one APEX sampling run."""

    atol: float = 0.0078
    rtol: float = 0.05
    err_exp: float = 0.5
    grow: float = 1.35
    shrink: float = 0.5
    e_hi: float = 1.0
    e_lo: float = 0.25
    max_order: int = 3
    use_spatial: bool = True
    spatial_topk: float = 0.15
    last_error: float = 0.0
    last_curvature: float = 0.0
    last_velocity: float = 0.0
    last_difficulty: float = 1.0
    last_phase: str = "reconstruction"
    step_scale: float = 1.0
    history_x0: list = field(default_factory=list)
    accepted: int = 0
    rejected_soft: int = 0

    @staticmethod
    def phase_name(progress: float) -> str:
        if progress < 0.28:
            return "structure"
        if progress < 0.72:
            return "reconstruction"
        return "refinement"

    def recommend_order(self, progress: float, hist_len: int) -> int:
        self.last_phase = self.phase_name(progress)
        cap = int(self.max_order) if self.last_phase == "reconstruction" else 2
        return min(cap, max(1, hist_len))

    def normalized_error(self, x_hi: torch.Tensor, x_lo: torch.Tensor, x: torch.Tensor) -> float:
        diff = (x_hi.float() - x_lo.float()).reshape(x.shape[0], -1).norm(dim=-1)
        scale = self.atol + self.rtol * x.float().reshape(x.shape[0], -1).norm(dim=-1).clamp(min=1e-6)
        self.last_error = float(diff.div(scale).mean().item())
        return self.last_error

    def spatial_difficulty(self, x0_cur: torch.Tensor, x0_prev: torch.Tensor | None) -> float:
        """Top-k / median ratio of the per-pixel ‖Δx0‖², mapped to [0.5, 3]."""
        if not self.use_spatial or x0_prev is None:
            return 1.0
        flat = (x0_cur.float() - x0_prev.float()).pow(2).mean(dim=1).reshape(x0_cur.shape[0], -1)
        k = max(1, int(self.spatial_topk * flat.shape[-1]))
        topv, _ = flat.topk(k, dim=-1)
        med = flat.median(dim=-1).values.clamp(min=1e-8)
        score = float((topv.mean(dim=-1) / med).mean().item())
        return max(0.5, min(3.0, 0.5 + score))

    def update_difficulty(
        self,
        *,
        error: float,
        curvature: float,
        velocity: float,
        spatial: float,
        progress: float,
    ) -> float:
        wp = {"structure": 0.85, "reconstruction": 1.25, "refinement": 1.0}[self.phase_name(progress)]
        d = 1.0 + 0.85 * error + 0.45 * curvature + 0.35 * velocity + 0.25 * (spatial - 1.0) + 0.15 * wp
        d = float(max(0.35, min(4.0, d)))
        self.last_difficulty = d
        self.last_curvature = float(curvature)
        self.last_velocity = float(velocity)
        if error > self.e_hi:
            self.step_scale = max(0.35, self.step_scale * self.shrink)
            self.rejected_soft += 1
        elif error < self.e_lo:
            self.step_scale = min(2.0, self.step_scale * self.grow)
        else:
            factor = float(np.clip((0.5 / max(error, 1e-6)) ** self.err_exp, 0.6, 1.4))
            self.step_scale = float(np.clip(self.step_scale * factor, 0.35, 2.0))
        self.accepted += 1
        return d


@dataclass
class ApexStepResult:
    x_next: torch.Tensor
    state: SolverState
    controller: ApexController
    error: float
    order_used: int
    used_safe_blend: bool = False


def _safe_blend(x_hi: torch.Tensor, x_lo: torch.Tensor, err: float, e_hi: float) -> torch.Tensor:
    """Pull a high-error step toward the first-order update (at most 85% of the way)."""
    w = float(np.clip((err - e_hi) / max(err, 1e-6), 0.0, 0.85))
    return (1.0 - w) * x_hi + w * x_lo


def _rms(x: torch.Tensor) -> float:
    return float(x.float().norm().item()) / math.sqrt(max(x.numel(), 1))


def vp_apex_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor | float,
    alpha_bar_next: torch.Tensor | float,
    state: SolverState,
    controller: ApexController,
    progress: float = 0.5,
) -> ApexStepResult:
    """Embedded order-1 vs phase-order VP step from one ``x0`` prediction (no extra NFE); updates ``state``."""
    order = controller.recommend_order(progress, hist_len=state.order + 1)
    step = {"x": x, "x0_pred": x0_pred, "alpha_bar_cur": alpha_bar_cur, "alpha_bar_next": alpha_bar_next}
    x_lo, _ = vp_dpmpp_update(**step, state=SolverState(max_order=1), order=1)
    x_hi, state = vp_dpmpp_update(**step, state=state, order=order)
    err = controller.normalized_error(x_hi, x_lo, x)

    hist = controller.history_x0
    vel = curv = 0.0
    if hist:
        delta = x0_pred.float() - hist[-1].float()
        vel = _rms(delta)
        if len(hist) >= 2:
            curv = _rms(delta - (hist[-1].float() - hist[-2].float()))
    spatial = controller.spatial_difficulty(x0_pred, hist[-1] if hist else None)
    controller.update_difficulty(error=err, curvature=curv, velocity=vel, spatial=spatial, progress=progress)
    hist.append(x0_pred.detach())
    if len(hist) > 4:
        hist.pop(0)

    used_safe = err > controller.e_hi and order > 1
    x_next = _safe_blend(x_hi, x_lo, err, controller.e_hi) if used_safe else x_hi
    return ApexStepResult(
        x_next=x_next.to(dtype=x.dtype),
        state=state,
        controller=controller,
        error=err,
        order_used=order,
        used_safe_blend=used_safe,
    )


def flow_apex_update(
    *,
    x: torch.Tensor,
    velocity: torch.Tensor,
    s_cur: float,
    s_next: float,
    state: SolverState,
    controller: ApexController,
    progress: float = 0.5,
) -> ApexStepResult:
    """Flow APEX: Euler vs DPM++(2M) embedded error from the same velocity (no extra NFE); updates ``state``."""
    x_lo = x + velocity * (float(s_next) - float(s_cur))
    x_hi, state = flow_dpmpp_2m_update(x=x, velocity=velocity, s_cur=s_cur, s_next=s_next, state=state)
    err = controller.normalized_error(x_hi, x_lo, x)
    controller.update_difficulty(error=err, curvature=0.0, velocity=_rms(velocity), spatial=1.0, progress=progress)
    used_safe = err > controller.e_hi
    x_next = _safe_blend(x_hi, x_lo, err, controller.e_hi) if used_safe else x_hi
    return ApexStepResult(
        x_next=x_next.to(dtype=x.dtype),
        state=state,
        controller=controller,
        error=err,
        order_used=2,
        used_safe_blend=used_safe,
    )
