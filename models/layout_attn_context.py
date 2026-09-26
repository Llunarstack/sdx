"""Thread-local layout attention state for Dense Diffusion soft-masks.

Set once per denoise step from ``sample_loop``; ``CrossAttention`` reads it so we
do not need to plumb kwargs through every DiTBlock.
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


def get_layout_attn() -> dict[str, Any] | None:
    return _layout_attn_ctx.get()


__all__ = ["set_layout_attn", "clear_layout_attn", "get_layout_attn"]
