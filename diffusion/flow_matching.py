"""
Linear **rectified-flow** training objective (Liu et al. 2022; SD3 recipe, Esser et al. 2024).

Uses ``x_t = (1 - s) x_0 + s \\epsilon`` with ``s = t / (T-1)`` and discrete ``t \\in [0, T-1]`` for the DiT timestep
embedding. The network output is matched to the velocity ``v^* = \\epsilon - x_0`` (same channels as latents).
Works for any latent rank (images ``(B, C, H, W)`` or video ``(B, C, T, H, W)``).

Sampling: ``GaussianDiffusion.sample_loop(..., flow_matching_sample=True)`` integrates in ``s`` on the same index grid,
with the same resolution shift (``flow_shift`` + ``flow_shift_ref_hw``) used here in training.
``ConsistencyFlowLoss`` adds the velocity-consistency term for few-step distillation of a trained flow.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from diffusion.schedules import time_shift


def sample_flow_time(
    n: int,
    *,
    logit_normal: bool = True,
    ln_mean: float = 0.0,
    ln_std: float = 1.0,
    shift: float = 1.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Draw ``n`` flow times ``s \\in (0, 1)`` (``s = 1`` is pure noise).

    ``logit_normal`` uses SD3's ``sigmoid(N(ln_mean, ln_std))`` density, which concentrates training on
    the mid-noise band where velocity is hardest to predict; otherwise ``s ~ U(0, 1)``. ``shift > 1``
    applies the SD3/Flux resolution shift ``s' = shift*s / (1 + (shift-1)*s)``, moving mass toward noise
    for large latents (more pixels keep more signal at a given ``s``). Pick it per batch from the token count
    with ``diffusion.schedules.sd3_resolution_shift`` or ``flux_resolution_shift`` and sample with the same rule.
    """
    if logit_normal:
        s = torch.sigmoid(torch.randn(int(n), device=device) * float(ln_std) + float(ln_mean))
    else:
        s = torch.rand(int(n), device=device)
    return time_shift(s, float(shift)) if float(shift) != 1.0 else s


def flow_time_to_index(s: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Snap flow times ``s`` to the integer ``t`` grid the sampler uses (``s = t / (T-1)``)."""
    return (s.float() * float(max(int(num_timesteps) - 1, 1))).round().long()


def shift_timestep_indices(t_idx: torch.Tensor, num_timesteps: int, shift: float) -> torch.Tensor:
    """Apply the SD3 resolution shift to already-sampled indices (so any ``timestep_sample_mode`` can be shifted)."""
    if float(shift) == 1.0:
        return t_idx
    denom = float(max(int(num_timesteps) - 1, 1))
    return flow_time_to_index(time_shift(t_idx.float() / denom, float(shift)), num_timesteps).to(t_idx.dtype)


def flow_matching_per_sample_losses(
    model: torch.nn.Module,
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    num_timesteps: int,
    model_kwargs: dict,
    *,
    t_idx: torch.Tensor | None = None,
    logit_normal: bool = False,
    ln_mean: float = 0.0,
    ln_std: float = 1.0,
    shift: float = 1.0,
    spectral_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Per-row velocity loss ``(B,)`` between ``model(x_t, t)`` and ``epsilon - x0``.

    ``t_idx`` given: used as is (the caller samples and shifts it). Otherwise ``s`` comes from
    :func:`sample_flow_time` (``logit_normal`` / ``shift``) and is snapped to the index grid so the timestep embedding
    and the interpolation agree exactly. ``spectral_kwargs`` (``low_sigma`` / ``high_sigma`` / ``tau_power``) swaps the
    MSE for the frequency-weighted loss of ``diffusion.spectral_sfp``.
    """
    if x0.shape != epsilon.shape:
        raise ValueError("x0 and epsilon must match shape")
    if t_idx is None:
        s = sample_flow_time(
            x0.shape[0], logit_normal=logit_normal, ln_mean=ln_mean, ln_std=ln_std, shift=shift, device=x0.device
        )
        t_idx = flow_time_to_index(s, num_timesteps)
    denom = float(max(int(num_timesteps) - 1, 1))
    st = (t_idx.float() / denom).view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    x_t = (1.0 - st) * x0 + st * epsilon
    out = model(x_t, t_idx, **model_kwargs)
    if out.shape != x0.shape and out.shape[1] > x0.shape[1]:
        out = out[:, : x0.shape[1]]
    target = epsilon - x0
    if spectral_kwargs:
        from diffusion.spectral_sfp import spectral_sfp_per_sample_loss

        return spectral_sfp_per_sample_loss(out, target, t_idx, int(num_timesteps), **spectral_kwargs)
    return (out - target).pow(2).flatten(1).mean(dim=1)


class ConsistencyFlowLoss:
    """
    Consistency regularisation on a rectified flow (Consistency Models, Song et al. 2023, adapted to velocity).

    The student's velocity at ``s`` is regressed onto an EMA teacher's velocity at ``s + delta`` on the same straight
    path ``x_s = (1 - s) x0 + s eps``; on a straight flow both equal ``eps - x0``, so agreement straightens the
    student's trajectories and makes few-step sampling accurate.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        *,
        consistency_weight: float = 1.0,
        ema_decay: float = 0.999,
        num_timesteps: int = 1000,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.consistency_weight = float(consistency_weight)
        self.ema_decay = float(ema_decay)
        self.num_timesteps = int(num_timesteps)

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        """Move the teacher toward the student (call after each optimizer step)."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.mul_(self.ema_decay).add_(s_param, alpha=1.0 - self.ema_decay)

    def compute(
        self,
        x0: torch.Tensor,
        epsilon: torch.Tensor,
        model_kwargs: dict[str, Any],
        *,
        s: torch.Tensor | None = None,
        delta_t: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Per-sample loss ``(B,)`` under ``loss`` and its mean under ``loss_mean``; ``s`` is SD3 logit-normal by default."""
        if s is None:
            s = sample_flow_time(x0.shape[0], logit_normal=True, device=x0.device)
        s = s.clamp(0.0, 1.0 - delta_t)
        s_next = s + delta_t
        shape = (-1,) + (1,) * (x0.dim() - 1)
        x_s = (1.0 - s.view(shape)) * x0 + s.view(shape) * epsilon
        x_next = (1.0 - s_next.view(shape)) * x0 + s_next.view(shape) * epsilon

        pred_student = self.student(x_s, flow_time_to_index(s, self.num_timesteps), **model_kwargs)
        with torch.no_grad():
            pred_teacher = self.teacher(x_next, flow_time_to_index(s_next, self.num_timesteps), **model_kwargs)
        c = x0.shape[1]
        loss = (pred_student[:, :c] - pred_teacher[:, :c]).pow(2).flatten(1).mean(dim=1) * self.consistency_weight
        return {"loss": loss, "loss_mean": loss.mean(), "s": s}


__all__ = [
    "ConsistencyFlowLoss",
    "flow_matching_per_sample_losses",
    "flow_time_to_index",
    "sample_flow_time",
    "shift_timestep_indices",
]
