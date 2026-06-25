"""
Dense Diffusion–style cross-attention biasing for box layouts.

Training-free: boosts attention between spatial patches and their regional prompts.
"""

from __future__ import annotations

from typing import Any

import torch


def bias_cross_attention(
    attn_logits: torch.Tensor,
    *,
    region_masks: torch.Tensor,
    step_index: int,
    plan: Any,
) -> torch.Tensor:
    """
    Apply layout bias to cross-attention logits ``(B, H, N, L)``.

    ``region_masks``: ``(R, 1, H, W)`` normalized 0–1 per region.
    When token layout is unavailable, boosts patches inside each box uniformly.
    """
    if plan is None or step_index not in set(getattr(plan, "enforce_steps", ())):
        return attn_logits
    strength = float(getattr(plan, "strength", 0.85))
    B, H, N, L = attn_logits.shape
    # flatten spatial: assume N = H_lat * W_lat
    hw = region_masks.shape[-2] * region_masks.shape[-1]
    if N != hw:
        return attn_logits
    masks = region_masks[:, 0].reshape(region_masks.shape[0], -1)  # R, N
    R = masks.shape[0]
    if R == 0 or L < R + 1:
        return attn_logits
    # Assume tokens 1..R align with regions (after global token 0)
    bias = torch.zeros(B, 1, N, L, device=attn_logits.device, dtype=attn_logits.dtype)
    for ri in range(R):
        tok = min(ri + 1, L - 1)
        m = masks[ri].view(1, 1, N, 1)
        bias[:, :, :, tok : tok + 1] += m * strength
    if getattr(plan, "backward_guidance", True):
        return attn_logits + bias
    return attn_logits - bias * 0.25
