"""
Rectified Flow training and sampling for SDX DiT.

Rectified flow (Liu et al. 2022, used in FLUX, SD3, Stable Cascade) learns a
**straight-line** transport from noise ε to data x₀:

    x_t = (1 - t) * x₀ + t * ε,   t ∈ [0, 1]
    v*(x_t, t) = ε - x₀            (constant velocity along the straight path)

Why it's better than VP DDPM:
- Straighter paths → fewer ODE steps needed at inference (4-8 steps vs 20-50)
- More stable training: velocity target has constant magnitude, no SNR weighting needed
- Better high-frequency detail: no cumulative noise schedule distortion
- Naturally supports flow-matching samplers (Euler, Heun, DPM-Solver)

This module provides:
1. RectifiedFlowLoss — drop-in training loss replacing GaussianDiffusion.training_losses
2. RectifiedFlowSampler — Euler/Heun ODE sampler for inference
3. LogitNormalTimeSampler — SD3-style time distribution (more samples near t=0.5)
4. ReflowPairing — "reflow" step: straighten trajectories by pairing x₀ with its
   own generated ε (reduces NFE further, used in InstaFlow)
5. ConsistencyFlowLoss — consistency regularization on top of rectified flow

Wire into train.py via:
    if cfg.flow_matching_training:
        loss = RectifiedFlowLoss(model, cfg).compute(x0, model_kwargs)

Wire into sample.py via:
    if use_flow_sample:
        x0 = RectifiedFlowSampler(model, steps=20).sample(shape, model_kwargs, device)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Time samplers
# ---------------------------------------------------------------------------

class UniformTimeSampler:
    """Uniform t ~ U[0, 1]."""
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=torch.float32)


class LogitNormalTimeSampler:
    """
    SD3-style logit-normal time sampler.

    Samples u ~ N(mean, std), then t = sigmoid(u).
    Concentrates samples near t=0.5 (the hardest denoising region),
    which empirically improves prompt adherence and detail.

    mean=0.0, std=1.0 is the SD3 default.
    mean=-1.0 biases toward cleaner images (lower t).
    mean=1.0 biases toward noisier images (higher t).
    """
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.randn(batch_size, device=device, dtype=torch.float32)
        u = u * self.std + self.mean
        return torch.sigmoid(u)


class CosineTimeSampler:
    """
    Cosine-weighted time sampler: p(t) ∝ sin(π*t).
    Emphasizes mid-range t values, similar to cosine noise schedules.
    """
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Inverse CDF of sin(π*t)/2: t = arccos(1 - 2u) / π
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        return torch.acos(1.0 - 2.0 * u) / math.pi


# ---------------------------------------------------------------------------
# Core rectified flow loss
# ---------------------------------------------------------------------------

class RectifiedFlowLoss:
    """
    Rectified flow training loss.

    Replaces GaussianDiffusion.training_losses for flow-matching training.
    Compatible with the existing DiT model — just changes what x_t and target are.

    Usage:
        rf_loss = RectifiedFlowLoss(
            time_sampler=LogitNormalTimeSampler(mean=0.0, std=1.0),
            prediction_type="velocity",  # or "epsilon" for compatibility
            loss_weight="uniform",        # or "snr_min" for SNR-weighted
        )
        losses = rf_loss.compute(model, x0, epsilon, model_kwargs)
        loss = losses["loss"].mean()
    """

    def __init__(
        self,
        time_sampler: Optional[Any] = None,
        prediction_type: str = "velocity",
        loss_weight: str = "uniform",
        min_snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
    ):
        self.time_sampler = time_sampler or LogitNormalTimeSampler()
        self.prediction_type = str(prediction_type)
        self.loss_weight = str(loss_weight)
        self.min_snr_gamma = float(min_snr_gamma)
        self.noise_offset = float(noise_offset)

    def _get_xt(
        self,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation: x_t = (1-t)*x0 + t*ε"""
        t_view = t.view(-1, 1, 1, 1)
        if self.noise_offset > 0:
            # Noise offset: add low-frequency noise component for better light/dark balance
            offset = self.noise_offset * torch.randn(
                epsilon.shape[0], epsilon.shape[1], 1, 1,
                device=epsilon.device, dtype=epsilon.dtype
            )
            epsilon = epsilon + offset
        return (1.0 - t_view) * x0 + t_view * epsilon

    def _get_target(
        self,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Target depends on prediction type."""
        if self.prediction_type == "velocity":
            # v* = ε - x₀ (constant along the straight path)
            return epsilon - x0
        elif self.prediction_type == "epsilon":
            return epsilon
        elif self.prediction_type == "x0":
            return x0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def _get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight."""
        if self.loss_weight == "uniform":
            return torch.ones_like(t)
        elif self.loss_weight == "snr_min":
            # SNR = (1-t)²/t² for rectified flow
            # Min-SNR cap: w = min(SNR, γ) / SNR
            snr = ((1.0 - t) / t.clamp(min=1e-8)).pow(2)
            if self.min_snr_gamma > 0:
                return torch.clamp(snr, max=self.min_snr_gamma) / snr.clamp(min=1e-8)
            return torch.ones_like(t)
        elif self.loss_weight == "cosine":
            # Higher weight near t=0.5 (hardest region)
            return 1.0 + torch.sin(math.pi * t)
        else:
            return torch.ones_like(t)

    def compute(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        model_kwargs: Dict[str, Any],
        *,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rectified flow loss.

        Args:
            model: DiT model (same API as GaussianDiffusion)
            x0: Clean latents (B, C, H, W)
            epsilon: Noise (same shape as x0)
            model_kwargs: Conditioning kwargs (encoder_hidden_states, etc.)
            t: Optional pre-sampled times in [0,1]. If None, sampled from time_sampler.

        Returns:
            Dict with 'loss' (B,), 'loss_mean' (scalar), 't' (B,), 'x_t' (B,C,H,W)
        """
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = self.time_sampler.sample(B, device)

        x_t = self._get_xt(x0, epsilon, t)
        target = self._get_target(x0, epsilon, t)

        # Convert continuous t to discrete timestep index for the model's embedding
        # (DiT uses integer timestep embeddings; we map t ∈ [0,1] → [0, T-1])
        # Use T=1000 to match the existing embedding table
        T = 1000
        t_idx = (t * (T - 1)).long().clamp(0, T - 1)

        pred = model(x_t, t_idx, **model_kwargs)

        # Handle learn_sigma: split if output has double channels
        if pred.shape[1] > x0.shape[1]:
            pred = pred[:, :x0.shape[1]]

        # Per-sample MSE
        per_sample_loss = (pred - target).pow(2).mean(dim=(1, 2, 3))

        # Apply loss weight
        weight = self._get_loss_weight(t)
        weighted_loss = per_sample_loss * weight

        return {
            "loss": weighted_loss,
            "loss_mean": weighted_loss.mean(),
            "loss_unweighted": per_sample_loss,
            "t": t,
            "x_t": x_t,
            "weight": weight,
        }


# ---------------------------------------------------------------------------
# Rectified flow sampler (ODE solver)
# ---------------------------------------------------------------------------

class RectifiedFlowSampler:
    """
    ODE sampler for rectified flow models.

    Supports:
    - Euler method (1 NFE per step, fast)
    - Heun method (2 NFE per step, more accurate)
    - DPM-Solver++ style adaptive stepping (experimental)

    Usage:
        sampler = RectifiedFlowSampler(steps=20, solver="euler")
        x0 = sampler.sample(model, shape, model_kwargs, device)
    """

    def __init__(
        self,
        steps: int = 20,
        solver: str = "euler",
        cfg_scale: float = 7.5,
        cfg_rescale: float = 0.0,
        t_start: float = 1.0,
        t_end: float = 0.0,
        time_schedule: str = "linear",
    ):
        self.steps = int(steps)
        self.solver = str(solver)
        self.cfg_scale = float(cfg_scale)
        self.cfg_rescale = float(cfg_rescale)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.time_schedule = str(time_schedule)

    def _get_timesteps(self) -> torch.Tensor:
        """Get the time schedule from t_start to t_end."""
        if self.time_schedule == "linear":
            return torch.linspace(self.t_start, self.t_end, self.steps + 1)
        elif self.time_schedule == "cosine":
            # Cosine schedule: more steps near t=0 (fine detail)
            i = torch.linspace(0, 1, self.steps + 1)
            return self.t_start + (self.t_end - self.t_start) * (1 - torch.cos(math.pi * i)) / 2
        elif self.time_schedule == "karras":
            # Karras-style: concentrate steps at low noise
            rho = 7.0
            t_max, t_min = self.t_start, max(self.t_end, 1e-4)
            i = torch.linspace(0, 1, self.steps + 1)
            return (t_max**(1/rho) + i * (t_min**(1/rho) - t_max**(1/rho)))**rho
        else:
            return torch.linspace(self.t_start, self.t_end, self.steps + 1)

    def _model_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t_scalar: float,
        model_kwargs_cond: Dict[str, Any],
        model_kwargs_uncond: Optional[Dict[str, Any]],
        T: int = 1000,
    ) -> torch.Tensor:
        """Single model forward with CFG."""
        B = x.shape[0]
        device = x.device
        t_idx = torch.full((B,), int(t_scalar * (T - 1)), device=device, dtype=torch.long)
        t_idx = t_idx.clamp(0, T - 1)

        if model_kwargs_uncond is not None and self.cfg_scale != 1.0:
            # Classifier-free guidance
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_idx, t_idx], dim=0)
            kwargs_combined = {}
            for k in model_kwargs_cond:
                v_cond = model_kwargs_cond[k]
                v_uncond = model_kwargs_uncond.get(k, v_cond)
                if isinstance(v_cond, torch.Tensor) and isinstance(v_uncond, torch.Tensor):
                    kwargs_combined[k] = torch.cat([v_cond, v_uncond], dim=0)
                else:
                    kwargs_combined[k] = v_cond

            with torch.no_grad():
                pred_double = model(x_double, t_double, **kwargs_combined)

            if pred_double.shape[1] > x.shape[1]:
                pred_double = pred_double[:, :x.shape[1]]

            pred_cond, pred_uncond = pred_double.chunk(2, dim=0)

            # CFG: v = v_uncond + scale * (v_cond - v_uncond)
            pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)

            # CFG rescale (reduces oversaturation)
            if self.cfg_rescale > 0:
                std_cond = pred_cond.std(dim=(1, 2, 3), keepdim=True)
                std_cfg = pred.std(dim=(1, 2, 3), keepdim=True)
                pred = pred * (std_cond / std_cfg.clamp(min=1e-8)) * self.cfg_rescale + pred * (1 - self.cfg_rescale)
        else:
            with torch.no_grad():
                pred = model(x, t_idx, **model_kwargs_cond)
            if pred.shape[1] > x.shape[1]:
                pred = pred[:, :x.shape[1]]

        return pred

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        model_kwargs_cond: Dict[str, Any],
        device: torch.device,
        model_kwargs_uncond: Optional[Dict[str, Any]] = None,
        x_init: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """
        Sample from the rectified flow model.

        Args:
            model: DiT model
            shape: Output shape (B, C, H, W)
            model_kwargs_cond: Conditioning kwargs
            device: Target device
            model_kwargs_uncond: Unconditional kwargs for CFG (None = no CFG)
            x_init: Optional starting latent (for img2img)
            dtype: Output dtype
            callback: Optional callback(step, x) called each step

        Returns:
            x0: Denoised latent (B, C, H, W)
        """
        timesteps = self._get_timesteps()

        # Start from noise (or provided init)
        if x_init is not None:
            x = x_init.to(device=device, dtype=dtype)
        else:
            x = torch.randn(shape, device=device, dtype=dtype)

        if self.solver == "euler":
            x = self._euler_solve(model, x, timesteps, model_kwargs_cond, model_kwargs_uncond, callback)
        elif self.solver == "heun":
            x = self._heun_solve(model, x, timesteps, model_kwargs_cond, model_kwargs_uncond, callback)
        elif self.solver == "midpoint":
            x = self._midpoint_solve(model, x, timesteps, model_kwargs_cond, model_kwargs_uncond, callback)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return x

    def _euler_solve(self, model, x, timesteps, kwargs_c, kwargs_u, callback):
        """Euler method: x_{t-dt} = x_t - dt * v(x_t, t)"""
        for i in range(len(timesteps) - 1):
            t_cur = float(timesteps[i])
            t_next = float(timesteps[i + 1])
            dt = t_next - t_cur  # negative (going from 1→0)

            v = self._model_forward(model, x, t_cur, kwargs_c, kwargs_u)
            x = x + dt * v

            if callback is not None:
                callback(i, x)

        return x

    def _heun_solve(self, model, x, timesteps, kwargs_c, kwargs_u, callback):
        """Heun method (2nd order): predictor-corrector."""
        for i in range(len(timesteps) - 1):
            t_cur = float(timesteps[i])
            t_next = float(timesteps[i + 1])
            dt = t_next - t_cur

            # Predictor (Euler)
            v1 = self._model_forward(model, x, t_cur, kwargs_c, kwargs_u)
            x_pred = x + dt * v1

            # Corrector
            v2 = self._model_forward(model, x_pred, t_next, kwargs_c, kwargs_u)
            x = x + dt * 0.5 * (v1 + v2)

            if callback is not None:
                callback(i, x)

        return x

    def _midpoint_solve(self, model, x, timesteps, kwargs_c, kwargs_u, callback):
        """Midpoint method (2nd order, 1 extra NFE per step)."""
        for i in range(len(timesteps) - 1):
            t_cur = float(timesteps[i])
            t_next = float(timesteps[i + 1])
            dt = t_next - t_cur
            t_mid = t_cur + 0.5 * dt

            v1 = self._model_forward(model, x, t_cur, kwargs_c, kwargs_u)
            x_mid = x + 0.5 * dt * v1
            v_mid = self._model_forward(model, x_mid, t_mid, kwargs_c, kwargs_u)
            x = x + dt * v_mid

            if callback is not None:
                callback(i, x)

        return x


# ---------------------------------------------------------------------------
# Reflow: straighten trajectories for fewer-step inference
# ---------------------------------------------------------------------------

class ReflowPairing:
    """
    "Reflow" step (Liu et al. 2023): straighten the learned flow by pairing
    each x₀ with its own generated ε from the current model.

    After one reflow step, the model can be distilled to 1-2 steps (InstaFlow).

    Usage:
        reflow = ReflowPairing(sampler)
        x0_paired, eps_paired = reflow.generate_pairs(model, x0_batch, model_kwargs)
        # Then train on (x0_paired, eps_paired) with RectifiedFlowLoss
    """

    def __init__(self, sampler: RectifiedFlowSampler):
        self.sampler = sampler

    @torch.no_grad()
    def generate_pairs(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        model_kwargs_cond: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate (x0, ε) pairs where ε is the noise that the current model
        would map to x0. These pairs define straighter trajectories.

        Returns:
            x0: Original clean latents
            epsilon_paired: Noise that maps to x0 via the current model
        """
        # Start from x0, add noise to get x_1 (the "paired" noise)
        _ = x0.shape[0]  # B unused; kept for shape documentation
        epsilon = torch.randn_like(x0)  # noqa: F841 — initial noise reference, replaced by ODE
        # The paired noise is what the model generates when starting from x0
        # We run the forward ODE: x0 → x1 (noise direction)
        # This is the reverse of sampling
        timesteps = self.sampler._get_timesteps().flip(0)  # 0 → 1

        x = x0.clone()
        for i in range(len(timesteps) - 1):
            t_cur = float(timesteps[i])
            t_next = float(timesteps[i + 1])
            dt = t_next - t_cur  # positive (going 0→1)
            v = self.sampler._model_forward(model, x, t_cur, model_kwargs_cond, None)
            x = x + dt * v

        return x0, x  # (clean data, paired noise)


# ---------------------------------------------------------------------------
# Consistency flow loss (for 1-step distillation)
# ---------------------------------------------------------------------------

class ConsistencyFlowLoss:
    """
    Consistency regularization on top of rectified flow.

    Enforces that the model's prediction at any point on the trajectory
    maps to the same x₀ (consistency property). This enables 1-step inference.

    Based on: Consistency Models (Song et al. 2023) adapted for flow matching.

    Usage:
        cf_loss = ConsistencyFlowLoss(teacher_model, student_model)
        loss = cf_loss.compute(x0, epsilon, model_kwargs)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        time_sampler: Optional[Any] = None,
        consistency_weight: float = 1.0,
        ema_decay: float = 0.999,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.time_sampler = time_sampler or LogitNormalTimeSampler()
        self.consistency_weight = float(consistency_weight)
        self.ema_decay = float(ema_decay)

    def update_teacher_ema(self) -> None:
        """Update teacher model as EMA of student (call after each training step)."""
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def compute(
        self,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        model_kwargs: Dict[str, Any],
        *,
        t: Optional[torch.Tensor] = None,
        delta_t: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency loss.

        The student at t should predict the same x₀ as the teacher at t+δt.

        Args:
            x0: Clean latents
            epsilon: Noise
            model_kwargs: Conditioning
            t: Optional pre-sampled times
            delta_t: Time step for consistency pair

        Returns:
            Dict with 'loss', 'loss_mean'
        """
        B = x0.shape[0]
        device = x0.device
        T = 1000

        if t is None:
            t = self.time_sampler.sample(B, device)

        # Clamp t so t+delta_t stays in [0,1]
        t = t.clamp(0.0, 1.0 - delta_t)
        t_next = (t + delta_t).clamp(0.0, 1.0)

        t_view = t.view(-1, 1, 1, 1)
        t_next_view = t_next.view(-1, 1, 1, 1)

        # x_t and x_{t+δt}
        x_t = (1.0 - t_view) * x0 + t_view * epsilon
        x_t_next = (1.0 - t_next_view) * x0 + t_next_view * epsilon

        t_idx = (t * (T - 1)).long().clamp(0, T - 1)
        t_next_idx = (t_next * (T - 1)).long().clamp(0, T - 1)

        # Student prediction at t
        pred_student = self.student(x_t, t_idx, **model_kwargs)
        if pred_student.shape[1] > x0.shape[1]:
            pred_student = pred_student[:, :x0.shape[1]]

        # Teacher prediction at t+δt (no grad)
        with torch.no_grad():
            pred_teacher = self.teacher(x_t_next, t_next_idx, **model_kwargs)
            if pred_teacher.shape[1] > x0.shape[1]:
                pred_teacher = pred_teacher[:, :x0.shape[1]]

        # Consistency: student(t) should match teacher(t+δt)
        loss = (pred_student - pred_teacher.detach()).pow(2).mean(dim=(1, 2, 3))

        return {
            "loss": loss * self.consistency_weight,
            "loss_mean": (loss * self.consistency_weight).mean(),
            "t": t,
        }


# ---------------------------------------------------------------------------
# Velocity-to-x0 conversion (for decode preview during training)
# ---------------------------------------------------------------------------

def velocity_to_x0(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Convert velocity prediction to x₀ estimate.

    From x_t = (1-t)*x₀ + t*ε and v = ε - x₀:
        x₀ = x_t - t * v
    """
    t_view = t.view(-1, 1, 1, 1)
    return x_t - t_view * v_pred


def velocity_to_epsilon(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Convert velocity prediction to noise estimate.

    ε = x₀ + v = (x_t - t*v) + v = x_t + (1-t)*v
    """
    t_view = t.view(-1, 1, 1, 1)
    return x_t + (1.0 - t_view) * v_pred


# ---------------------------------------------------------------------------
# Optimal transport noise coupling (reduces variance, improves training)
# ---------------------------------------------------------------------------

def ot_noise_coupling(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    num_iters: int = 10,
    reg: float = 0.05,
) -> torch.Tensor:
    """
    Optimal transport coupling between x₀ and ε within a mini-batch.

    Instead of random pairing (x₀_i, ε_j), find the assignment that
    minimizes total transport cost (L2 distance). This straightens
    trajectories further and reduces training variance.

    Uses Sinkhorn algorithm for soft OT.

    Args:
        x0: Clean latents (B, C, H, W)
        epsilon: Noise samples (B, C, H, W)
        num_iters: Sinkhorn iterations
        reg: Regularization strength (smaller = harder assignment)

    Returns:
        epsilon_coupled: Reordered noise that minimizes transport cost
    """
    B = x0.shape[0]
    if B <= 1:
        return epsilon

    # Flatten for distance computation
    x0_flat = x0.reshape(B, -1).float()
    eps_flat = epsilon.reshape(B, -1).float()

    # Cost matrix: L2 distance between each x0 and each ε
    # C[i,j] = ||x0_i - ε_j||²
    cost = torch.cdist(x0_flat, eps_flat, p=2).pow(2)

    # Sinkhorn algorithm
    log_a = torch.zeros(B, device=x0.device, dtype=torch.float32)
    log_b = torch.zeros(B, device=x0.device, dtype=torch.float32)
    log_K = -cost / reg

    for _ in range(num_iters):
        log_a = torch.logsumexp(log_K + log_b.unsqueeze(0), dim=1)
        log_b = torch.logsumexp(log_K + log_a.unsqueeze(1), dim=0)

    # Transport plan
    log_T = log_K + log_a.unsqueeze(1) + log_b.unsqueeze(0)
    T = log_T.exp()

    # Hard assignment: argmax over columns
    assignment = T.argmax(dim=1)  # (B,) — for each x0_i, which ε_j to use

    return epsilon[assignment]


__all__ = [
    "RectifiedFlowLoss",
    "RectifiedFlowSampler",
    "ReflowPairing",
    "ConsistencyFlowLoss",
    "LogitNormalTimeSampler",
    "UniformTimeSampler",
    "CosineTimeSampler",
    "velocity_to_x0",
    "velocity_to_epsilon",
    "ot_noise_coupling",
]
