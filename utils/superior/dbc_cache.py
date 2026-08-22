"""
**DBCache-style** block cache helpers (Cache-DiT inspired).

When CFG batches cond+uncond in one forward (2×B), fingerprints should use only the
conditional half so cache keys stay stable across guidance branches.

See https://github.com/vipshop/cache-dit (``enable_separate_cfg``).
"""

from __future__ import annotations

import torch

from utils.superior.block_cache import BlockDiTCache


def begin_block_cache_forward(
    cache: BlockDiTCache,
    t_emb: torch.Tensor,
    x: torch.Tensor,
    *,
    cfg_split: bool = False,
) -> None:
    """
    Begin a block-cache step; optionally fingerprint only the cond half of a CFG batch.
    """
    te = t_emb
    xs = x
    if bool(cfg_split) and te.shape[0] >= 2 and te.shape[0] % 2 == 0:
        half = te.shape[0] // 2
        te = te[:half]
        xs = xs[:half]
    cache.begin_forward(BlockDiTCache.fingerprint_from_tensors(te, xs))


def build_dbc_block_cache(
    threshold: float,
    recompute_every: int,
    *,
    separate_cfg: bool = True,
) -> tuple[BlockDiTCache, bool]:
    """Return ``(cache, cfg_split_flag)`` for diffusion sampling."""
    from utils.superior.block_cache import BlockCacheConfig

    if float(threshold) <= 0.0:
        return None, False  # type: ignore[return-value]
    cache = BlockDiTCache(
        BlockCacheConfig(
            rel_l1_threshold=float(threshold),
            recompute_every=max(1, int(recompute_every)),
        )
    )
    return cache, bool(separate_cfg)


__all__ = ["begin_block_cache_forward", "build_dbc_block_cache"]
