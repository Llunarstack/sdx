"""
**SLG** — Skip Layer Guidance (Stable Diffusion 3.5 / Diffusers guider).

Training-free: ``guided = cfg + slg_scale * (cond_full - cond_skipped)``.

Requires a third forward with ``skip_blocks`` on the DiT (see ``models/dit_text.py``).
"""

from __future__ import annotations

from typing import Tuple

import torch


def default_skip_block_indices(depth: int, *, fraction: float = 0.55) -> Tuple[int, ...]:
    """Heuristic middle blocks to skip (SD3.5 uses ~7,8,9 on 28 blocks)."""
    d = max(1, int(depth))
    mid = int(round(float(fraction) * (d - 1)))
    return tuple(sorted({max(0, mid - 1), mid, min(d - 1, mid + 1)}))


def slg_combine(
    out_cond: torch.Tensor,
    out_uncond: torch.Tensor,
    out_cond_skipped: torch.Tensor,
    *,
    cfg_scale: float,
    slg_scale: float = 2.8,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """Apply CFG then add skip-layer guidance shift."""
    delta = out_cond - out_uncond
    if float(cfg_rescale) > 0.0:
        sig = delta.std() + 1e-8
        delta = delta / max(sig / float(cfg_rescale), 1.0)
    cfg_pred = out_uncond + float(cfg_scale) * delta
    slg_delta = out_cond - out_cond_skipped
    return cfg_pred + float(slg_scale) * slg_delta


def parse_skip_blocks(spec: str, *, depth: int = 28) -> Tuple[int, ...]:
    """Parse ``7,8,9`` or ``auto`` into block indices."""
    s = str(spec or "").strip().lower()
    if not s or s == "none":
        return ()
    if s == "auto":
        return default_skip_block_indices(depth)
    out: list[int] = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return tuple(sorted(set(out)))


__all__ = [
    "default_skip_block_indices",
    "parse_skip_blocks",
    "slg_combine",
]
