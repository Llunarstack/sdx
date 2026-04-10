"""Rank multiple samples or skip expensive steps using ``perceptual_proxies``."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .perceptual_proxies import combined_quality_proxy

Tensor = torch.Tensor


def rank_samples_by_proxy(rgb01: Tensor) -> List[Tuple[int, float]]:
    """
    Return ``(batch_index, score)`` sorted by descending ``combined_quality_proxy``.

    ``rgb01`` is ``(B, 3, H, W)``.
    """
    scores = combined_quality_proxy(rgb01)
    indexed = [(int(i), float(scores[i].item())) for i in range(scores.shape[0])]
    indexed.sort(key=lambda t: t[1], reverse=True)
    return indexed


def gate_by_proxy_threshold(
    rgb01: Tensor,
    *,
    threshold: float,
    scores: Tensor | None = None,
) -> Tensor:
    """
    Boolean mask ``(B,)``: True where proxy score ``>= threshold``.

    Pass ``scores`` if you already computed ``combined_quality_proxy`` to avoid duplicate work.
    """
    s = scores if scores is not None else combined_quality_proxy(rgb01)
    return s >= float(threshold)


def best_of_n_index(rgb01_batch: Sequence[Tensor]) -> int:
    """
    Pick index of the tensor with highest mean proxy (each element ``(1,3,H,W)`` or ``(3,H,W)``).

    Raises if empty.
    """
    if not rgb01_batch:
        raise ValueError("empty batch sequence")
    best_i = 0
    best_score = float("-inf")
    for i, t in enumerate(rgb01_batch):
        x = t if t.dim() == 4 else t.unsqueeze(0)
        sc = float(combined_quality_proxy(x)[0].item())
        if sc > best_score:
            best_score = sc
            best_i = i
    return best_i
