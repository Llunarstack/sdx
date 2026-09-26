"""
In-gen scene cuts — Seedance/Wan multi-scene inside one clip.

Competitors sell 30s with built-in cuts. Our retrieve pipeline is multi-segment;
this module plans hard cuts + tempo shifts *within* a frame sequence and can
insert cut flashes / speed ramps so one segment reads as multi-shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["SceneCutPlan", "CutBeat", "plan_scene_cuts", "apply_scene_cuts"]


@dataclass(slots=True)
class CutBeat:
    frame_index: int
    kind: str = "hard"  # hard|smash|tempo
    note: str = ""


@dataclass(slots=True)
class SceneCutPlan:
    cuts: list[CutBeat] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_scene_cuts(
    prompt: str,
    *,
    frame_count: int,
    fps: float = 24.0,
    max_cuts: int = 4,
) -> SceneCutPlan:
    text = (prompt or "").lower()
    n = max(1, int(frame_count))
    cuts: list[CutBeat] = []
    notes: list[str] = []

    # Explicit multi-shot language
    markers = 0
    for token in ("then ", "cut to", "wide shot", "close-up", "next,", "meanwhile", "later"):
        if token in text:
            markers += 1
    target = min(max_cuts, max(markers, 1 if "multi" in text or "montage" in text else 0))
    if target <= 0 and n >= int(fps * 8):
        # Long clip → at least one mid cut for cinematic pacing (Wan multi-shot default)
        target = 1
        notes.append("auto_mid_cut_long_clip")

    if target <= 0:
        return SceneCutPlan(notes=["no_cuts"])

    # Evenly space cuts away from edges
    span = n - 4
    for i in range(target):
        fi = 2 + int((i + 1) * span / (target + 1))
        kind = "smash" if "montage" in text or "flash" in text else "hard"
        cuts.append(CutBeat(frame_index=fi, kind=kind, note=f"beat_{i}"))
    notes.append(f"planned_cuts={len(cuts)}")
    return SceneCutPlan(cuts=cuts, notes=notes)


def apply_scene_cuts(
    frame_paths: list[Path],
    plan: SceneCutPlan,
    *,
    flash_frames: int = 1,
) -> list[Path]:
    """
    Emphasize planned cuts: optional 1-frame flash + micro hold on either side.
    Does not change frame count.
    """
    paths = list(frame_paths)
    if not plan.cuts or len(paths) < 6:
        return paths
    for cut in plan.cuts:
        i = int(np.clip(cut.frame_index, 1, len(paths) - 2))
        if cut.kind == "smash" and flash_frames > 0:
            rgb = read_frame_rgb(paths[i])
            flash = np.clip(rgb.astype(np.float32) * 1.8 + 40, 0, 255).astype(np.uint8)
            save_frame_rgb(paths[i], flash)
        # Micro hold: duplicate previous onto i-1 for snappier cut perception
        if i >= 2:
            prev = read_frame_rgb(paths[i - 2])
            save_frame_rgb(paths[i - 1], prev)
    return paths
