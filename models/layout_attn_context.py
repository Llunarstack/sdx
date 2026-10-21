"""Context-local Dense Diffusion layout state (a ``ContextVar``).

``set_layout_attn`` activates it around a model call (see ``diffusion.gaussian_diffusion``); ``CrossAttention``
and ``TACA`` read it through ``layout_cross_attn_bias``, so no kwarg has to be plumbed through every block.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch

_layout_attn_ctx: ContextVar[dict[str, Any] | None] = ContextVar("sdx_layout_attn", default=None)


def set_layout_attn(
    *,
    plan: Any | None = None,
    region_masks: torch.Tensor | None = None,
    step_index: int = 0,
    cond_batch_size: int | None = None,
    num_patch_tokens: int | None = None,
) -> None:
    """Activate layout bias for subsequent CrossAttention forwards."""
    if plan is None or region_masks is None:
        _layout_attn_ctx.set(None)
        return
    _layout_attn_ctx.set(
        {
            "plan": plan,
            "region_masks": region_masks,
            "step_index": int(step_index),
            "cond_batch_size": cond_batch_size,
            "num_patch_tokens": num_patch_tokens,
        }
    )


def clear_layout_attn() -> None:
    _layout_attn_ctx.set(None)


def layout_cross_attn_bias(
    batch: int, n_query: int, n_text: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor | None:
    """Additive ``(B, 1, N, L)`` cross-attention bias for the active layout step, else ``None``."""
    ctx = _layout_attn_ctx.get()
    if ctx is None:
        return None
    from frontier.attention import bias_cross_attention

    zeros = torch.zeros(batch, 1, n_query, n_text, device=device, dtype=dtype)
    bias = bias_cross_attention(zeros, **ctx)
    return None if bias is zeros else bias


__all__ = ["set_layout_attn", "clear_layout_attn", "layout_cross_attn_bias"]
