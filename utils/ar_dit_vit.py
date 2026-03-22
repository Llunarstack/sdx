"""
**DiT block-AR ↔ ViT scorer** bridge.

DiT can use ``num_ar_blocks`` (0 / 2 / 4) for block-causal self-attention (see ``docs/AR.md``).
Images from an **AR-trained** checkpoint often have different failure modes than full
bidirectional DiT. The ViT quality model can take a small **AR regime vector** so scores
are calibrated per training/inference layout.

JSONL fields (any one):

- ``num_ar_blocks``, ``dit_num_ar_blocks``, ``ar_blocks`` — integer ``0``, ``2``, or ``4``
  (``-1`` or missing → **unknown** one-hot).

See also: ``models/attention.py`` ``create_block_causal_mask_2d``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

# One-hot layout: [full (0), ar_2x2 (2), ar_4x4 (4), unknown]
AR_COND_DIM: int = 4

_VALID_AR: frozenset[int] = frozenset({0, 2, 4})


def normalize_num_ar_blocks(value: Any) -> int:
    """
    Return ``0``, ``2``, ``4``, or ``-1`` (unknown / invalid).
    """
    if value is None:
        return -1
    try:
        v = int(value)
    except (TypeError, ValueError):
        return -1
    if v in _VALID_AR:
        return v
    return -1


def parse_num_ar_blocks_from_row(row: Mapping[str, Any]) -> int:
    for key in ("num_ar_blocks", "dit_num_ar_blocks", "ar_blocks"):
        if key in row:
            return normalize_num_ar_blocks(row.get(key))
    return -1


def ar_conditioning_vector(num_ar_blocks: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Single-row vector ``(AR_COND_DIM,)``: one-hot for DiT AR regime + unknown bucket.
    """
    v = torch.zeros(AR_COND_DIM, device=device, dtype=dtype)
    if num_ar_blocks == 0:
        v[0] = 1.0
    elif num_ar_blocks == 2:
        v[1] = 1.0
    elif num_ar_blocks == 4:
        v[2] = 1.0
    else:
        v[3] = 1.0
    return v


def batch_ar_conditioning(
    values: Sequence[int],
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """``(B, AR_COND_DIM)`` from a list of normalized ``num_ar_blocks`` (use ``-1`` for unknown)."""
    rows = [ar_conditioning_vector(int(v), device=device, dtype=dtype) for v in values]
    return torch.stack(rows, dim=0)


def default_unknown_ar_batch(batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """All-unknown regime (index 3)."""
    u = torch.zeros(batch_size, AR_COND_DIM, device=device, dtype=dtype)
    u[:, 3] = 1.0
    return u
