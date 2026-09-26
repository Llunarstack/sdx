"""Physics assist repair — damp impossible slot jumps when physics score is low."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permanence import extract_slots, match_slots
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["PhysicsRepairReport", "apply_physics_repair"]


@dataclass(slots=True)
class PhysicsRepairReport:
    score_before: float
    score_after: float = 1.0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def apply_physics_repair(
    frame_paths: list[Path] | list[str],
    *,
    strength: float = 0.45,
    trigger_below: float = 0.55,
) -> PhysicsRepairReport:
    """Temporal blend frames where primary-slot velocity flips / jumps hard.

    Training-free: when physics_gate score is poor, blend mid frames toward
    neighbors to kill anti-gravity pops and teleport discontinuities.
    """
    from .physics_gate import score_physics_invariance

    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    before = score_physics_invariance(paths)
    report = PhysicsRepairReport(score_before=float(before.score))
    if len(paths) < 3 or before.score >= float(trigger_below):
        report.score_after = float(before.score)
        report.notes.append("physics_ok_or_short")
        return report

    s = float(np.clip(strength, 0.05, 0.85))
    repaired = 0
    # Read once
    frames = [read_frame_rgb(p).astype(np.float32) for p in paths]
    h, w = frames[0].shape[:2]
    prev_slots = extract_slots(frames[0].astype(np.uint8), k=6)
    for i in range(1, len(frames) - 1):
        slots = extract_slots(frames[i].astype(np.uint8), k=6)
        pairs, dis, _app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.35)
        jump = bool(dis)
        for pi, cj in pairs:
            if pi >= len(prev_slots) or cj >= len(slots):
                continue
            a, b = prev_slots[pi], slots[cj]
            dy = (a.cy - b.cy) / max(float(h), 1.0)
            dx = (a.cx - b.cx) / max(float(w), 1.0)
            # >18% of frame diagonal in one step → teleport / physics break
            if (dy * dy + dx * dx) ** 0.5 > 0.18:
                jump = True
                break
        if before.anti_gravity_events or before.velocity_flip_penalties or before.jump_discontinuities or jump:
            blended = (1.0 - s) * frames[i] + (s * 0.5) * frames[i - 1] + (s * 0.5) * frames[i + 1]
            frames[i] = np.clip(blended, 0, 255)
            repaired += 1
        prev_slots = slots

    if repaired:
        for p, fr in zip(paths, frames):
            save_frame_rgb(p, fr.astype(np.uint8))
    after = score_physics_invariance(paths)
    report.score_after = float(after.score)
    report.repaired = repaired
    report.notes.append(f"blended_frames={repaired}")
    return report
