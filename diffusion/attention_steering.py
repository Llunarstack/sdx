"""
Attention Steering (AST) — training-free inference-time attention manipulation.

Based on the observation that DiT layers have a hierarchical structure:
- Early layers (~0–33%): instance/subject tokens dominate.
- Middle layers (~33–66%): background and spatial layout.
- Late layers (~66–100%): attributes, style, fine detail.

AST injects a guidance signal into the cross-attention maps at each step using:

    A_hat = f_norm(A * exp(beta_t * G * (M - A)))

where:
- A is the raw attention map (B, H, N, L)
- M is a target mask (B, 1, N, L) — which patches should attend to which tokens
- G is a per-layer gate (scalar or (L,)) — which tokens to steer
- beta_t is a step-dependent strength that decays over denoising

This is a complement to Holy Grail, not a replacement. Holy Grail schedules
*scale* (how much CFG/control/adapter); AST steers *where* attention lands.

Usage (inference only — no training changes):
    from diffusion.attention_steering import AttentionSteerer
    steerer = AttentionSteerer(total_steps=40, beta_max=2.0)
    # In your sampling loop, after getting attn_weights from the model:
    steered = steerer.steer(attn_weights, step_index=i, token_mask=mask)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ASTConfig:
    """Configuration for Attention Steering."""

    # Maximum steering strength (beta at step 0).
    beta_max: float = 2.0
    # Minimum steering strength (beta at final step).
    beta_min: float = 0.0
    # Decay schedule: "cosine" or "linear".
    decay: str = "cosine"
    # Which layer range to steer: "all", "early", "middle", "late".
    layer_range: str = "all"
    # Clamp attention values after steering to prevent numerical issues.
    clamp_min: float = 1e-6


def _beta_at_step(step: int, total: int, cfg: ASTConfig) -> float:
    """Compute steering strength beta_t for the current step."""
    if total <= 1:
        return float(cfg.beta_max)
    p = float(step) / float(total - 1)  # 0 → 1 over denoising
    if cfg.decay == "cosine":
        w = 0.5 * (1.0 + math.cos(math.pi * p))
    else:
        w = 1.0 - p
    return float(cfg.beta_min) + (float(cfg.beta_max) - float(cfg.beta_min)) * w


def _layer_in_range(layer_idx: int, total_layers: int, layer_range: str) -> bool:
    """Return True if this layer should be steered."""
    if layer_range == "all" or total_layers <= 0:
        return True
    p = float(layer_idx) / float(max(total_layers - 1, 1))
    if layer_range == "early":
        return p <= 1.0 / 3.0
    if layer_range == "middle":
        return 1.0 / 3.0 < p <= 2.0 / 3.0
    if layer_range == "late":
        return p > 2.0 / 3.0
    return True


def steer_attention(
    attn: torch.Tensor,
    *,
    target_mask: torch.Tensor,
    beta: float,
    token_gate: Optional[torch.Tensor] = None,
    clamp_min: float = 1e-6,
) -> torch.Tensor:
    """
    Apply one AST step to a cross-attention map.

    Args:
        attn: Raw attention logits or probabilities ``(B, H, N, L)``.
        target_mask: Desired attention target ``(B, 1, N, L)`` or ``(B, H, N, L)``,
            values in ``[0, 1]``. Patches that should attend to a token get value 1.
        beta: Steering strength for this step.
        token_gate: Optional per-token gate ``(L,)`` or ``(B, L)`` — which tokens
            to steer (1 = steer, 0 = leave alone). None = steer all tokens.
        clamp_min: Floor for attention values after steering.

    Returns:
        Steered attention of the same shape as ``attn``.
    """
    if beta == 0.0:
        return attn

    a = attn.to(dtype=torch.float32)
    m = target_mask.to(device=a.device, dtype=torch.float32)

    # Broadcast mask to (B, H, N, L) if needed.
    if m.dim() == 3:
        m = m.unsqueeze(1)
    if m.shape[1] == 1 and a.shape[1] > 1:
        m = m.expand_as(a)

    # Per-token gate: shape (B, 1, 1, L) for broadcasting.
    if token_gate is not None:
        g = token_gate.to(device=a.device, dtype=torch.float32)
        if g.dim() == 1:
            g = g.view(1, 1, 1, -1)
        elif g.dim() == 2:
            g = g.unsqueeze(1).unsqueeze(2)
    else:
        g = torch.ones(1, 1, 1, a.shape[-1], device=a.device, dtype=torch.float32)

    # AST formula: A_hat = f_norm(A * exp(beta * G * (M - A)))
    guidance = beta * g * (m - a)
    a_steered = a * torch.exp(guidance)

    # Re-normalise over token dimension (L) to keep it a valid distribution.
    a_steered = a_steered.clamp(min=clamp_min)
    a_steered = a_steered / (a_steered.sum(dim=-1, keepdim=True) + 1e-8)

    return a_steered.to(dtype=attn.dtype)


class AttentionSteerer:
    """
    Stateful wrapper that applies AST across a full sampling run.

    Tracks the current step and applies the correct beta decay automatically.

    Example::

        steerer = AttentionSteerer(total_steps=40, cfg=ASTConfig(beta_max=2.0))
        for i, t in enumerate(timesteps):
            # ... model forward with return_attn=True ...
            attn_steered = steerer.steer(attn_weights, step_index=i, target_mask=mask)
            # Use attn_steered to re-weight the model output (see sample.py integration).
    """

    def __init__(
        self,
        total_steps: int,
        cfg: Optional[ASTConfig] = None,
    ):
        self.total_steps = int(total_steps)
        self.cfg = cfg or ASTConfig()

    def beta(self, step_index: int) -> float:
        return _beta_at_step(step_index, self.total_steps, self.cfg)

    def steer(
        self,
        attn: torch.Tensor,
        *,
        step_index: int,
        target_mask: torch.Tensor,
        layer_idx: int = 0,
        total_layers: int = 0,
        token_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Steer attention for one layer at one step.

        Args:
            attn: ``(B, H, N, L)`` attention map.
            step_index: Current denoising step (0 = most noisy).
            target_mask: ``(B, 1, N, L)`` or ``(B, H, N, L)`` target.
            layer_idx: Index of this layer (for layer_range filtering).
            total_layers: Total number of layers (for layer_range filtering).
            token_gate: Optional per-token gate.

        Returns:
            Steered attention (same shape as input).
        """
        if not _layer_in_range(layer_idx, total_layers, self.cfg.layer_range):
            return attn
        b = self.beta(step_index)
        if b == 0.0:
            return attn
        return steer_attention(
            attn,
            target_mask=target_mask,
            beta=b,
            token_gate=token_gate,
            clamp_min=self.cfg.clamp_min,
        )


__all__ = ["ASTConfig", "AttentionSteerer", "steer_attention"]
