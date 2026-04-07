"""
DiT block-AR <-> ViT scorer bridge (SDX DiT, Meta/Facebook DiT-style checkpoints, JSONL).

SDX DiT uses ``num_ar_blocks`` in ``{0, 2, 4}`` for block-causal self-attention (see ``docs/AR.md``).
Facebook Research DiT (``facebookresearch/DiT``) is full bidirectional by default - treat as
``num_ar_blocks=0`` unless you train a fork with the same mask API.

The ViT quality model fuses a 4-D one-hot AR regime with caption stats so scores match the
generator layout. Use the helpers below to read AR from checkpoints, tag manifests, and
keep training / inference / ViT aligned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import torch

# One-hot layout: [full (0), ar_2x2 (2), ar_4x4 (4), unknown]
AR_COND_DIM: int = 4

_VALID_AR: frozenset[int] = frozenset({0, 2, 4})

# Human-readable names for logs / manifests / UIs
AR_REGIME_NAMES: Dict[int, str] = {
    0: "full_bidirectional",
    2: "block_ar_2x2",
    4: "block_ar_4x4",
    -1: "unknown",
}

# Order for flat JSONL key scan (first hit wins)
_AR_FLAT_KEYS: Tuple[str, ...] = (
    "num_ar_blocks",
    "dit_num_ar_blocks",
    "ar_blocks",
    "generator_num_ar_blocks",
)

_NESTED_KEYS: Tuple[str, ...] = ("dit_config", "train_config", "checkpoint_config", "model_config")


def normalize_num_ar_blocks(value: Any) -> int:
    """
    Return ``0``, ``2``, ``4``, or ``-1`` (unknown / invalid).
    Accepts int-like strings (e.g. JSON ``"2"``).
    """
    if value is None:
        return -1
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return -1
    try:
        v = int(value)
    except (TypeError, ValueError):
        return -1
    if v in _VALID_AR:
        return v
    return -1


def ar_regime_label(num_ar_blocks: int) -> str:
    """Stable slug: ``full_bidirectional``, ``block_ar_2x2``, ``block_ar_4x4``, ``unknown``."""
    return AR_REGIME_NAMES.get(int(num_ar_blocks), AR_REGIME_NAMES[-1])


def parse_num_ar_blocks_from_row(row: Mapping[str, Any], *, max_nested_depth: int = 2) -> int:
    """
    Parse AR regime from a manifest or metadata dict.

    Checks flat keys first, then optional nested config blobs (depth-limited).
    """
    return _parse_num_ar_blocks_from_row_depth(row, depth=0, max_depth=max_nested_depth)


def _parse_num_ar_blocks_from_row_depth(
    row: Mapping[str, Any], *, depth: int, max_depth: int
) -> int:
    for key in _AR_FLAT_KEYS:
        if key in row:
            return normalize_num_ar_blocks(row.get(key))
    if depth >= max_depth:
        return -1
    for nk in _NESTED_KEYS:
        sub = row.get(nk)
        if isinstance(sub, Mapping):
            v = _parse_num_ar_blocks_from_row_depth(sub, depth=depth + 1, max_depth=max_depth)
            if v != -1:
                return v
    return -1


def _extract_num_ar_from_config_obj(cfg: Any) -> int:
    if cfg is None:
        return -1
    if isinstance(cfg, dict):
        for k in _AR_FLAT_KEYS:
            if k in cfg:
                return normalize_num_ar_blocks(cfg[k])
        return -1
    for k in _AR_FLAT_KEYS:
        if hasattr(cfg, k):
            return normalize_num_ar_blocks(getattr(cfg, k, None))
    return -1


def num_ar_blocks_from_checkpoint_dict(ckpt: Mapping[str, Any]) -> int:
    """
    Read ``num_ar_blocks`` from a loaded SDX-style checkpoint dict (``torch.load``).

    Looks at ``ckpt["config"]`` (dict or dataclass-like). Returns ``-1`` if missing.
    """
    return _extract_num_ar_from_config_obj(ckpt.get("config"))


def read_num_ar_blocks_from_checkpoint(path: Union[str, Path]) -> int:
    """
    Load checkpoint from disk (CPU) and return ``num_ar_blocks`` for tagging manifests / ViT defaults.

    Does not instantiate DiT - safe for large checkpoints.
    """
    p = Path(path)
    if not p.is_file():
        return -1
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return -1
    if not isinstance(ckpt, dict):
        return -1
    v = num_ar_blocks_from_checkpoint_dict(ckpt)
    if v != -1:
        return v
    # Some exports store flat
    return normalize_num_ar_blocks(ckpt.get("num_ar_blocks"))


def tag_manifest_row_ar(
    row: Mapping[str, Any],
    num_ar_blocks: int,
    *,
    overwrite: bool = False,
    primary_key: str = "num_ar_blocks",
) -> Dict[str, Any]:
    """
    Copy *row* and set ``num_ar_blocks`` (and mirror ``dit_num_ar_blocks``) for ViT / digest tools.

    If *overwrite* is False, leaves existing valid AR fields unchanged.
    Unknown *num_ar_blocks* (not in 0,2,4) -> return unchanged copy.
    """
    out = dict(row)
    if num_ar_blocks not in _VALID_AR:
        return out
    current = parse_num_ar_blocks_from_row(out)
    if not overwrite and current != -1:
        return out
    v = int(num_ar_blocks)
    out[primary_key] = v
    out["dit_num_ar_blocks"] = v
    out["ar_regime"] = ar_regime_label(v)
    return out


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
    """``(B, AR_COND_DIM)`` from normalized ``num_ar_blocks`` (use ``-1`` for unknown)."""
    rows = [ar_conditioning_vector(int(v), device=device, dtype=dtype) for v in values]
    return torch.stack(rows, dim=0)


def default_unknown_ar_batch(batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
    """All-unknown regime (index 3)."""
    u = torch.zeros(batch_size, AR_COND_DIM, device=device, dtype=dtype)
    u[:, 3] = 1.0
    return u


def vit_text_total_dim(text_feat_dim: int, *, use_ar_conditioning: bool, ar_cond_dim: int = AR_COND_DIM) -> int:
    """Linear input width for ViT ``text_proj`` (caption stats + optional AR one-hot)."""
    t = int(text_feat_dim)
    return t + (int(ar_cond_dim) if use_ar_conditioning else 0)


def ar_bridge_summary_lines() -> str:
    """Help text for CLIs (DiT <-> ViT)."""
    return (
        "DiT num_ar_blocks: 0=full bidirectional (default Meta DiT-style), 2=2x2 block AR, 4=4x4 block AR. "
        "ViT uses a 4-D one-hot (full / ar2 / ar4 / unknown) fused with caption stats. "
        "Tag JSONL with num_ar_blocks or pass --default-num-ar-blocks on ViT infer."
    )


__all__ = [
    "AR_COND_DIM",
    "AR_REGIME_NAMES",
    "ar_bridge_summary_lines",
    "ar_conditioning_vector",
    "ar_regime_label",
    "batch_ar_conditioning",
    "default_unknown_ar_batch",
    "normalize_num_ar_blocks",
    "num_ar_blocks_from_checkpoint_dict",
    "parse_num_ar_blocks_from_row",
    "read_num_ar_blocks_from_checkpoint",
    "tag_manifest_row_ar",
    "vit_text_total_dim",
]

