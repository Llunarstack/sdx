"""InstantStyle-style reference token routing for DiT cross-attn.

Early transformer blocks tend to lock layout/subject; later blocks carry more
palette / brush / lighting. Injecting image tokens only into late blocks (and
optionally subtracting CLIP content) reduces subject bleed vs full-context concat.
"""

from __future__ import annotations

from typing import Any

import torch


def resolve_reference_block_start(
    *,
    n_blocks: int,
    style_mode: str = "full",
    block_start: int = -1,
) -> int:
    """First block index that receives reference tokens (inclusive)."""
    n = max(1, int(n_blocks))
    explicit = int(block_start)
    if explicit >= 0:
        return min(max(0, explicit), n)
    mode = str(style_mode or "full").strip().lower()
    if mode in ("full", "all", "ip", "ip-adapter", ""):
        return 0
    if mode in ("instantstyle", "style", "late", "b-lora-style"):
        # Last third ≈ InstantStyle / B-LoRA style blocks on DiT depth.
        return (n * 2) // 3
    if mode in ("style_mid", "layout_style", "mid"):
        return n // 2
    return 0


def concat_reference_tokens(
    text_emb: torch.Tensor,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Returns (text_without_ref, text_with_ref_or_None).

    When reference is disabled / scale<=0, with_ref is None.
    """
    ref_tok = kwargs.get("reference_tokens")
    if ref_tok is None:
        return text_emb, None
    rs = float(kwargs.get("reference_scale", 1.0) or 0.0)
    if rs <= 0.0:
        return text_emb, None

    rt = ref_tok.to(device=text_emb.device, dtype=text_emb.dtype)
    if rt.shape[0] != text_emb.shape[0]:
        if rt.shape[0] == 1:
            rt = rt.expand(text_emb.shape[0], -1, -1)
        else:
            raise ValueError(
                f"reference_tokens batch must match encoder_hidden_states batch ({rt.shape[0]} vs {text_emb.shape[0]})"
            )
    return text_emb, torch.cat([text_emb, rt * rs], dim=1)


def pick_block_text_emb(
    text_emb_plain: torch.Tensor,
    text_emb_with_ref: torch.Tensor | None,
    *,
    block_index: int,
    ref_block_start: int,
) -> torch.Tensor:
    if text_emb_with_ref is None or block_index < ref_block_start:
        return text_emb_plain
    return text_emb_with_ref


__all__ = [
    "concat_reference_tokens",
    "pick_block_text_emb",
    "resolve_reference_block_start",
]
