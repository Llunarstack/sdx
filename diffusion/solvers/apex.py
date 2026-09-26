"""
APEX — Adaptive Predictive EXtrapolation
========================================

Sampler + scheduler co-designed for VP diffusion and rectified flow.

Design (refined from classical adaptive ODE practice + diffusion-specific phases):

1. **Static schedule** (reproducible, batchable): three-phase density on noise
   progress — composition early, densest mid-band reconstruction, larger late
   refinement steps (Karras-like power warp as base).

2. **Embedded error estimate** (no extra NFE): compare DPM++ order-1 vs order-2+
   updates from the same ``x0`` prediction (DPM-Solver-12 style).

3. **Curvature / velocity**: track ‖Δx0‖ across steps for difficulty.

4. **Adaptive redistribute** (optional): after each accepted step, rebuild the
   remaining discrete timesteps from current index → 0 using difficulty so the
   NFE budget stays fixed while spacing adapts.

5. **Phase-aware order**: structure (≤2) → reconstruction (2–3) → refinement (2).

6. **Spatial difficulty** (optional scalar): top-k fraction of spatial ‖Δx0‖
   feeds the global difficulty score (not a per-pixel ODE).

References: Lu et al. DPM-Solver / DPM-Solver++ adaptive step-size; Karras EDM
σ schedules; Align-Your-Steps knot densities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from diffusion.solvers.base import SolverState
from diffusion.solvers.dpm_solver_pp import vp_dpmpp_update

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
    """Progress in [0, 1] noisy→clean. Higher = denser steps."""
    p = float(np.clip(progress, 0.0, 1.0))
    mid = math.exp(-((p - 0.42) ** 2) / (2 * 0.18**2))
    late = 0.35 * math.exp(-((p - 0.82) ** 2) / (2 * 0.12**2))
    early = 0.45 + 0.25 * (1.0 - p)
    return float(early + 1.35 * mid + late)


def _apex_density_curve(n: int, *, rho: float = 7.0) -> np.ndarray:
    n = max(1, int(n))
    # Optional Rust cdylib: tight-loop density (three exp per knot) with no
    # Python interpreter overhead. Rebuilt every accepted step in the adaptive
    # path, so it is on the sampling hot path. Falls back to NumPy below.
    try:
        from sdx_native.apex_schedule_native import maybe_apex_density_curve_rust

        result = maybe_apex_density_curve_rust(n, float(rho))
        if result is not None:
            return result
    except Exception:
        pass
    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    rho = float(max(rho, 1.0))
    karras = u ** (1.0 / rho)
    dens = np.array([phase_weight(float(ui)) for ui in u], dtype=np.float64)
    dens = dens * (0.65 + 0.35 * karras)
    dens = np.maximum(dens, 1e-8)
    dens /= dens.sum()
    return dens


def _nearest_lambda_indices(lam: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the closest ``lam`` entry per ``target`` (numpy.argmin tie-break).

    Optional Rust cdylib replaces the O(len(targets) * len(lam)) Python list-comp;
    falls back to NumPy when the library is not built.
    """
    try:
        from sdx_native.apex_schedule_native import maybe_apex_nearest_lambda_indices_rust

        result = maybe_apex_nearest_lambda_indices_rust(lam, targets)
        if result is not None:
            return result
    except Exception:
        pass
    return np.array([int(np.argmin(np.abs(lam - lt))) for lt in targets], dtype=np.int64)


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
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    if n > int(num_train):
        # Degenerate (more steps than training timesteps): distinct knots are
        # impossible, so fall back to an even spread rather than a zero run.
        return np.clip(np.round(np.linspace(num_train - 1, 0, n)).astype(np.int64), 0, num_train - 1)
    out = arr.copy()
    floors = np.arange(n - 1, -1, -1, dtype=np.int64)  # step j still needs n-1-j knots below it
    np.maximum(out, floors, out=out)
    for j in range(1, n):
        if out[j] >= out[j - 1]:
            out[j] = out[j - 1] - 1  # stays >= floors[j] because out[j-1] >= floors[j-1]
    out[-1] = 0
    return out.astype(np.int64)


def build_apex_vp_indices(
    num_train: int,
    num_infer: int,
    alpha_cumprod: np.ndarray,
    *,
    rho: float = 7.0,
) -> np.ndarray:
    """Static APEX VP schedule: descending integer indices (noisy → clean)."""
    T = int(num_train)
    N = max(2, int(num_infer))
    ac = np.asarray(alpha_cumprod, dtype=np.float64).reshape(-1)
    if ac.size < T:
        ac = np.linspace(0.9999, 1e-4, T) if ac.size == 0 else np.resize(ac, T)
    ac = np.clip(ac[:T], 1e-12, 1.0 - 1e-12)
    # λ(t) = ½ log(ᾱ/(1-ᾱ)); t=0 clean (high λ), t=T-1 noisy (low λ)
    lam = 0.5 * np.log(ac / (1.0 - ac))
    lam_noisy = float(lam[T - 1])
    lam_clean = float(lam[0])
    dens = _apex_density_curve(N - 1, rho=rho)
    dlam_total = lam_clean - lam_noisy
    if abs(dlam_total) < 1e-12:
        return np.linspace(T - 1, 0, N, dtype=np.int64)

    cum = np.concatenate([[0.0], np.cumsum(dens)])
    lam_targets = lam_noisy + cum * dlam_total
    idx = _nearest_lambda_indices(lam, lam_targets)
    idx[0] = max(int(idx[0]), 1)
    idx[-1] = 0
    return _enforce_desc(idx, T)[:N] if len(idx) >= N else _enforce_desc(np.linspace(T - 1, 0, N, dtype=np.int64), T)


def build_apex_flow_s_grid(
    num_steps: int,
    *,
    s_start: float = 1.0,
    s_end: float = 0.0,
    rho: float = 7.0,
) -> np.ndarray:
    """Continuous flow ``s`` grid with APEX three-phase density (s: 1→0)."""
    n = max(1, int(num_steps))
    dens = _apex_density_curve(n, rho=rho)
    cum = np.concatenate([[0.0], np.cumsum(dens)])
    grid = (float(s_start) + (float(s_end) - float(s_start)) * cum).astype(np.float64)
    # cumsum lands on 1±eps, which can push the final knot just past ``s_end``
    # (a small negative s for the default 1→0 grid). Pin both ends exactly.
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
    """Rebuild indices after ``t_cur`` down to 0 (length ``remaining_steps``)."""
    rem = max(1, int(remaining_steps))
    T = int(num_train)
    t0 = int(np.clip(t_cur, 1, T - 1))
    if rem == 1:
        return np.asarray([0], dtype=np.int64)
    # Every knot must be distinct and strictly below ``t_cur``, so only ``t0``
    # of them can fit. Returning the shorter feasible tail lets the caller keep
    # its existing schedule (it checks the length) instead of us handing back a
    # run of duplicate zeros or knots that step back *above* the current time.
    rem = min(rem, t0)
    if rem <= 1:
        return np.asarray([0], dtype=np.int64)
    dens = _apex_density_curve(rem - 1, rho=rho)
    d = float(max(difficulty, 0.25))
    boost = np.linspace(d, 1.0 / max(d, 1e-3), rem - 1)
    dens = dens * boost
    dens = dens / dens.sum()
    cum = np.concatenate([[0.0], np.cumsum(dens)])
    targets = t0 + cum * (0 - t0)
    idx = np.round(targets).astype(np.int64)
    idx[0] = min(int(idx[0]), t0 - 1)
    return _enforce_desc(idx, T)[-rem:]


@dataclass
class ApexController:
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

    def phase_name(self, progress: float) -> str:
        p = float(progress)
        if p < 0.28:
            return "structure"
        if p < 0.72:
            return "reconstruction"
        return "refinement"

    def recommend_order(self, progress: float, hist_len: int) -> int:
        phase = self.phase_name(progress)
        self.last_phase = phase
        if phase == "structure":
            return min(2, max(1, hist_len))
        if phase == "reconstruction":
            return min(int(self.max_order), max(1, hist_len))
        return min(2, max(1, hist_len))

    def normalized_error(self, x_hi: torch.Tensor, x_lo: torch.Tensor, x: torch.Tensor) -> float:
        diff = (x_hi.float() - x_lo.float()).reshape(x.shape[0], -1).norm(dim=-1)
        scale = self.atol + self.rtol * x.float().reshape(x.shape[0], -1).norm(dim=-1).clamp(min=1e-6)
        e = float(diff.div(scale).mean().item())
        self.last_error = e
        return e

    def spatial_difficulty(self, x0_cur: torch.Tensor, x0_prev: torch.Tensor | None) -> float:
        if not self.use_spatial or x0_prev is None:
            return 1.0
        d = (x0_cur.float() - x0_prev.float()).pow(2).mean(dim=1)
        flat = d.reshape(d.shape[0], -1)
        k = max(1, int(self.spatial_topk * flat.shape[-1]))
        topv, _ = flat.topk(k, dim=-1)
        med = flat.median(dim=-1).values.clamp(min=1e-8)
        score = float((topv.mean(dim=-1) / med).mean().item())
        return float(max(0.5, min(3.0, 0.5 + score)))

    def update_difficulty(
        self,
        *,
        error: float,
        curvature: float,
        velocity: float,
        spatial: float,
        progress: float,
    ) -> float:
        phase = self.phase_name(progress)
        wp = {"structure": 0.85, "reconstruction": 1.25, "refinement": 1.0}[phase]
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


def _clone_state(state: SolverState) -> SolverState:
    s = SolverState(max_order=state.max_order)
    s.model_outputs = [t.clone() for t in state.model_outputs]
    s.timesteps = list(state.timesteps)
    return s


def vp_apex_update(
    *,
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_cur: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    state: SolverState,
    controller: ApexController,
    progress: float = 0.5,
    eta: float = 0.0,
) -> ApexStepResult:
    """Embedded order-1 vs phase-order APEX step (single NFE already spent)."""
    st_lo = _clone_state(state)
    st_hi = _clone_state(state)
    order = controller.recommend_order(progress, hist_len=state.order + 1)

    x_lo, st_lo = vp_dpmpp_update(
        x=x,
        x0_pred=x0_pred,
        alpha_bar_cur=alpha_bar_cur,
        alpha_bar_next=alpha_bar_next,
        state=st_lo,
        order=1,
    )
    x_hi, st_hi = vp_dpmpp_update(
        x=x,
        x0_pred=x0_pred,
        alpha_bar_cur=alpha_bar_cur,
        alpha_bar_next=alpha_bar_next,
        state=st_hi,
        order=max(1, order),
    )
    err = controller.normalized_error(x_hi, x_lo, x)

    x0_prev = controller.history_x0[-1] if controller.history_x0 else None
    if x0_prev is not None:
        vel = float((x0_pred.float() - x0_prev.float()).norm().item()) / math.sqrt(max(x0_pred.numel(), 1))
    else:
        vel = 0.0
    if len(controller.history_x0) >= 2:
        v1 = controller.history_x0[-1].float() - controller.history_x0[-2].float()
        v2 = x0_pred.float() - controller.history_x0[-1].float()
        curv = float((v2 - v1).norm().item()) / math.sqrt(max(x0_pred.numel(), 1))
    else:
        curv = 0.0
    spatial = controller.spatial_difficulty(x0_pred, x0_prev)
    controller.update_difficulty(error=err, curvature=curv, velocity=vel, spatial=spatial, progress=progress)
    controller.history_x0.append(x0_pred.detach())
    if len(controller.history_x0) > 4:
        controller.history_x0.pop(0)

    used_safe = False
    if err > controller.e_hi and order > 1:
        w = float(np.clip((err - controller.e_hi) / max(err, 1e-6), 0.0, 0.85))
        x_next = (1.0 - w) * x_hi + w * x_lo
        used_safe = True
        out_state = st_lo if w > 0.5 else st_hi
    else:
        x_next = x_hi
        out_state = st_hi

    if float(eta) > 0.0:
        ab_c = alpha_bar_cur
        ab_n = alpha_bar_next
        if ab_c.ndim == 0:
            ab_c = ab_c.expand(x.shape[0])
            ab_n = ab_n.expand(x.shape[0])
        sigma = (
            float(eta)
            * (((1 - ab_n) / (1 - ab_c).clamp(min=1e-12)) * (1 - ab_c / ab_n.clamp(min=1e-12))).clamp(min=0).sqrt()
        )
        while sigma.ndim < x_next.ndim:
            sigma = sigma.view(*sigma.shape, *([1] * (x_next.ndim - sigma.ndim)))
        x_next = x_next + sigma.to(dtype=x_next.dtype, device=x_next.device) * torch.randn_like(x_next)

    return ApexStepResult(
        x_next=x_next.to(dtype=x.dtype),
        state=out_state,
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
    """Flow APEX: Euler vs multistep embedded error (same velocity / NFE)."""
    from diffusion.solvers.dpm_solver_pp import flow_dpmpp_2m_update

    ds = float(s_next) - float(s_cur)
    x_lo = x + velocity * ds
    st_hi = _clone_state(state)
    x_hi, st_hi = flow_dpmpp_2m_update(x=x, velocity=velocity, s_cur=s_cur, s_next=s_next, state=st_hi)
    err = controller.normalized_error(x_hi, x_lo, x)
    vel = float(velocity.float().norm().item()) / math.sqrt(max(velocity.numel(), 1))
    controller.update_difficulty(error=err, curvature=0.0, velocity=vel, spatial=1.0, progress=progress)
    used_safe = False
    if err > controller.e_hi:
        w = float(np.clip((err - controller.e_hi) / max(err, 1e-6), 0.0, 0.85))
        x_next = (1.0 - w) * x_hi + w * x_lo
        used_safe = True
    else:
        x_next = x_hi
    return ApexStepResult(
        x_next=x_next.to(dtype=x.dtype),
        state=st_hi,
        controller=controller,
        error=err,
        order_used=2,
        used_safe_blend=used_safe,
    )
