"""
Dense Diffusion–style cross-attention biasing for box layouts.

Training-free: boosts attention between spatial patches and their regional prompts
during the structure phase (the first ``inject_frac`` of steps).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["AttentionLayoutPlan", "bias_cross_attention", "build_attention_layout_plan", "region_masks_from_boxes"]


@dataclass
class AttentionLayoutPlan:
    region_names: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]
    enforce_steps: tuple[int, ...]
    strength: float = 0.85


def build_attention_layout_plan(
    regions: Sequence[Any],
    *,
    num_steps: int = 28,
    inject_frac: float = 0.4,
    strength: float = 0.85,
) -> AttentionLayoutPlan:
    """Plan from box-layout regions (objects with ``name`` and ``x1``…``y2``); enforce the first ``inject_frac`` steps."""
    names = tuple(str(getattr(r, "name", f"region_{i}")) for i, r in enumerate(regions))
    boxes = tuple(
        (
            float(getattr(r, "x1", 0)),
            float(getattr(r, "y1", 0)),
            float(getattr(r, "x2", 1)),
            float(getattr(r, "y2", 1)),
        )
        for r in regions
    )
    n_enforce = max(1, int(round(num_steps * inject_frac)))
    return AttentionLayoutPlan(
        region_names=names,
        boxes=boxes,
        enforce_steps=tuple(range(n_enforce)),
        strength=float(strength),
    )


def region_masks_from_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    latent_h: int,
    latent_w: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Hard ``(R, 1, H, W)`` masks from normalized ``(x1, y1, x2, y2)`` boxes."""
    masks = torch.zeros(len(boxes), 1, latent_h, latent_w, device=device, dtype=dtype)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        y0, x0 = int(y1 * latent_h), int(x1 * latent_w)
        masks[i, :, y0 : max(y0 + 1, int(y2 * latent_h)), x0 : max(x0 + 1, int(x2 * latent_w))] = 1.0
    return masks


def _patch_grid(n: int, mask_h: int, mask_w: int) -> tuple[int, int] | None:
    """Token grid ``(h, w)`` with ``h * w == n`` and the masks' aspect ratio."""
    if n == mask_h * mask_w:
        return mask_h, mask_w
    h = round(math.sqrt(n * mask_h / mask_w))
    if h <= 0 or n % h:
        return None
    return h, n // h


def bias_cross_attention(
    attn_logits: torch.Tensor,
    *,
    region_masks: torch.Tensor,
    step_index: int,
    plan: Any,
    cond_batch_size: int | None = None,
    num_patch_tokens: int | None = None,
) -> torch.Tensor:
    """
    Add layout bias to cross-attention logits ``(B, H, N, L)``.

    Text tokens are split into one contiguous chunk per region (plan order); chunk ``r`` gains
    ``strength * mask_r`` on the patch queries inside region ``r``. ``region_masks`` ``(R, 1, h, w)``
    are area-pooled onto the patch grid. Only the first ``cond_batch_size`` rows (the conditional
    half of a CFG batch) and the first ``num_patch_tokens`` queries (registers excluded) are biased.
    Returns ``attn_logits`` itself when the plan does not apply at this step.
    """
    if plan is None or step_index not in set(getattr(plan, "enforce_steps", ())):
        return attn_logits
    b, _, n_query, n_text = attn_logits.shape
    n_regions = int(region_masks.shape[0])
    n = min(int(num_patch_tokens or n_query), n_query)
    rows = min(int(cond_batch_size if cond_batch_size is not None else b), b)
    grid = _patch_grid(n, int(region_masks.shape[-2]), int(region_masks.shape[-1]))
    if n_regions == 0 or n_text < n_regions or rows <= 0 or grid is None:
        return attn_logits
    masks = F.adaptive_avg_pool2d(region_masks.to(attn_logits.device, attn_logits.dtype), grid).flatten(1)
    chunks = torch.zeros(n_regions, n_text, device=attn_logits.device, dtype=attn_logits.dtype)
    for r in range(n_regions):
        chunks[r, r * n_text // n_regions : (r + 1) * n_text // n_regions] = 1.0
    bias = float(getattr(plan, "strength", 0.85)) * (masks.t() @ chunks)
    out = attn_logits.clone()
    out[:rows, :, :n] += bias
    return out
