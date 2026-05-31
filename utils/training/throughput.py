"""Training/inference throughput helpers (no change to model math)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def encode_neg_and_style_embeddings(
    neg_caps: Sequence[str],
    styles: Sequence[str],
    *,
    style_embed_dim: int,
    encode_fn,
    batch_text_encode: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Encode negative and style caption lists; batched T5 when both are needed."""
    need_neg = any(n and str(n).strip() for n in neg_caps)
    need_style = bool(style_embed_dim) and any(s and str(s).strip() for s in styles)
    encoder_hidden_neg: Optional[torch.Tensor] = None
    style_embedding: Optional[torch.Tensor] = None
    if not need_neg and not need_style:
        return None, None

    if batch_text_encode and need_neg and need_style:
        neg_emb, style_emb = encode_text_multi_group(
            [[n or "" for n in neg_caps], [s or "" for s in styles]],
            encode_fn,
        )
        encoder_hidden_neg = neg_emb
        if style_emb is not None:
            style_embedding = style_emb.mean(dim=1)
    else:
        if need_neg:
            encoder_hidden_neg = encode_fn([n or "" for n in neg_caps])
        if need_style:
            style_embedding = encode_fn([s or "" for s in styles]).mean(dim=1)
    return encoder_hidden_neg, style_embedding


__all__ = [
    "create_adamw_optimizer",
    "encode_neg_and_style_embeddings",
    "encode_text_multi_group",
    "to_train_device",
]


def to_train_device(
    x: torch.Tensor,
    device: torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Move only when needed (skips work when CUDA prefetch already placed tensors)."""
    if x.device != device:
        x = x.to(device, non_blocking=device.type == "cuda")
    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype=dtype)
    return x


def create_adamw_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    fused: bool = True,
) -> torch.optim.AdamW:
    """AdamW with fused CUDA kernel when supported (same numerics, faster step)."""
    kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": betas}
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
        try:
            return torch.optim.AdamW(params, **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
    return torch.optim.AdamW(params, **kwargs)


def encode_text_multi_group(
    groups: Sequence[Optional[Sequence[str]]],
    encode_fn,
) -> List[Optional[torch.Tensor]]:
    """
    One ``encode_fn(captions)`` call for all non-empty caption groups.

    *encode_fn* must accept a list of strings and return ``(N, L, D)`` embeddings.
    Groups that are ``None`` or empty stay ``None`` in the output.
    """
    indexed: List[Tuple[int, List[str]]] = []
    for i, g in enumerate(groups):
        if not g:
            continue
        caps = [c or "" for c in g]
        if caps:
            indexed.append((i, caps))

    out: List[Optional[torch.Tensor]] = [None] * len(groups)
    if not indexed:
        return out

    flat: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    for i, caps in indexed:
        start = len(flat)
        flat.extend(caps)
        spans.append((i, start, len(flat)))

    emb = encode_fn(flat)
    for i, start, end in spans:
        out[i] = emb[start:end]
    return out
