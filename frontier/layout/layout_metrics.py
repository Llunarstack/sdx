"""
LAMIC layout quality metrics: Inclusion Ratio (IN-R), Fill Ratio (FI-R).

Cheap QA for box layouts before spending GPU on a full sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayoutQualityReport:
    inclusion_ratio: float
    fill_ratio: float
    overlap_ratio: float
    background_ratio: float
    notes: str


def score_layout_masks(
    region_masks: torch.Tensor,
    bg_mask: torch.Tensor,
    *,
    target_boxes_fill: float = 0.55,
) -> LayoutQualityReport:
    """
    Score latent-region masks.

    ``region_masks``: (R, 1, H, W), ``bg_mask``: (1, 1, H, W).
    """
    if region_masks.numel() == 0:
        return LayoutQualityReport(0.0, 0.0, 0.0, 1.0, "no regions")

    rm = region_masks.float().clamp(0, 1)
    bg = bg_mask.float().clamp(0, 1)
    r = int(rm.shape[0])

    union = rm.sum(dim=0, keepdim=True).clamp(0, 1)
    overlap = (rm.sum(dim=0, keepdim=True) - union.clamp(max=1.0)).clamp(min=0)
    overlap_ratio = float(overlap.mean().item())

    per_region_mean = [float(rm[i].mean().item()) for i in range(r)]
    inclusion_ratio = float(sum(1 for m in per_region_mean if m > 0.05) / max(1, r))
    fill_ratio = float(sum(per_region_mean) / max(1, r))
    background_ratio = float(bg.mean().item())

    notes_parts = []
    if fill_ratio < target_boxes_fill * 0.5:
        notes_parts.append("regions are very small; consider larger boxes")
    if overlap_ratio > 0.15:
        notes_parts.append("high overlap; check priority or feather")
    if background_ratio < 0.05:
        notes_parts.append("layout covers almost entire frame")

    return LayoutQualityReport(
        inclusion_ratio=inclusion_ratio,
        fill_ratio=fill_ratio,
        overlap_ratio=overlap_ratio,
        background_ratio=background_ratio,
        notes="; ".join(notes_parts) if notes_parts else "ok",
    )
