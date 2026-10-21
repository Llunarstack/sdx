"""Layout control: Omost canvas, coordinate prompts, LAMIC schedules, metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "OmostCanvas",
    "canvas_to_box_layout",
    "bind_coordinates_to_prompt",
    "RegionFusionSchedule",
    "fusion_weight_at_step",
    "LayoutQualityReport",
    "score_layout_masks",
]


def bind_coordinates_to_prompt(
    prompt: str,
    box: tuple[float, float, float, float],
    *,
    prefix: bool = True,
) -> str:
    """Insert ``<loc_x1_y1_x2_y2>`` token before prompt text."""
    x1, y1, x2, y2 = (round(float(v), 3) for v in box)
    token = f"<loc_{x1}_{y1}_{x2}_{y2}>"
    p = (prompt or "").strip()
    if token.lower() in p.lower():
        return p
    return f"{token} {p}" if prefix else f"{p} {token}"


@dataclass(frozen=True)
class RegionFusionSchedule:
    """Controls how much regions blend with each other over steps."""

    isolate_until_frac: float = 0.45
    curve: str = "cosine"  # linear | cosine
    min_isolation: float = 0.85  # 1 = full isolation early
    max_fusion: float = 1.0


def fusion_weight_at_step(
    step_index: int,
    total_steps: int,
    schedule: RegionFusionSchedule,
) -> float:
    """
    Return cross-region fusion factor in ``[0, 1]``.

    Low early (isolated), high late (fused). Multiply regional mask overlap
    blending by this factor when implementing LAMIC-style attention hooks.
    """
    if total_steps <= 1:
        return schedule.max_fusion
    p = step_index / (total_steps - 1)
    frac = schedule.isolate_until_frac
    if p <= frac:
        iso = schedule.min_isolation + (1.0 - schedule.min_isolation) * _ease(p / max(1e-6, frac), schedule.curve)
        return 1.0 - iso
    return schedule.max_fusion * _ease((p - frac) / max(1e-6, 1.0 - frac), schedule.curve)


def _ease(t: float, curve: str) -> float:
    return t if curve == "linear" else 0.5 * (1.0 - math.cos(math.pi * t))


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
    # Coverage beyond the first region at each pixel.
    overlap_ratio = float((rm.sum(dim=0) - 1.0).clamp(min=0).mean())
    per_region_mean = rm.flatten(1).mean(dim=1)
    inclusion_ratio = float((per_region_mean > 0.05).float().mean())
    fill_ratio = float(per_region_mean.mean())
    background_ratio = float(bg_mask.float().clamp(0, 1).mean())

    notes_parts: list[str] = []
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


# Omost uses 9 anchor boxes; we expose the same grid for LLM compatibility.
OMOST_GRID: dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.33, 0.33),
    "top": (0.33, 0.0, 0.66, 0.33),
    "top_right": (0.66, 0.0, 1.0, 0.33),
    "left": (0.0, 0.33, 0.33, 0.66),
    "center": (0.33, 0.33, 0.66, 0.66),
    "right": (0.66, 0.33, 1.0, 0.66),
    "bottom_left": (0.0, 0.66, 0.33, 1.0),
    "bottom": (0.33, 0.66, 0.66, 1.0),
    "bottom_right": (0.66, 0.66, 1.0, 1.0),
}


@dataclass
class LocalDescription:
    prompt: str
    box: tuple[float, float, float, float]
    name: str
    negative: str = ""
    priority: int = 5
    reference: str = ""


@dataclass
class OmostCanvas:
    """Virtual canvas agent (Omost-compatible API surface)."""

    global_description: str = ""
    global_negative: str = ""
    local_regions: list[LocalDescription] = field(default_factory=list)
    feather: int = 8

    def set_global_description(self, text: str, *, negative: str = "") -> None:
        self.global_description = (text or "").strip()
        if negative:
            self.global_negative = negative.strip()

    def add_local_description(
        self,
        prompt: str,
        *,
        box: tuple[float, float, float, float] | None = None,
        anchor: str | None = None,
        name: str | None = None,
        negative: str = "",
        priority: int = 5,
        reference: str = "",
    ) -> None:
        if anchor:
            key = anchor.strip().lower().replace("-", "_").replace(" ", "_")
            if key not in OMOST_GRID:
                raise ValueError(f"unknown anchor {anchor!r}; choose from {list(OMOST_GRID)}")
            b = OMOST_GRID[key]
        elif box is not None:
            b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        else:
            raise ValueError("provide box=(x1,y1,x2,y2) or anchor='center' etc.")
        idx = len(self.local_regions)
        self.local_regions.append(
            LocalDescription(
                prompt=(prompt or "").strip(),
                box=b,
                name=name or f"region_{idx}",
                negative=negative,
                priority=priority,
                reference=reference,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        regions: list[dict[str, Any]] = []
        for loc in self.local_regions:
            r: dict[str, Any] = {
                "name": loc.name,
                "box": list(loc.box),
                "prompt": loc.prompt,
                "priority": loc.priority,
            }
            if loc.negative:
                r["negative"] = loc.negative
            if loc.reference:
                r["reference"] = loc.reference
            regions.append(r)
        out: dict[str, Any] = {
            "global_prompt": self.global_description,
            "feather": self.feather,
            "regions": regions,
        }
        if self.global_negative:
            out["global_negative"] = self.global_negative
        return out


def canvas_to_box_layout(canvas: OmostCanvas) -> dict[str, Any]:
    """Export dict suitable for ``json.dump`` → ``--box-layout``."""
    if not canvas.local_regions:
        raise ValueError("canvas has no local descriptions")
    return canvas.to_dict()
