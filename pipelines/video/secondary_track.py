"""
Secondary track — props that must stay glued to the hero (phones, bags, weapons).

Competitors teleport handbags and phones. We track a secondary slot near the
primary subject and reinject when it pops off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permanence import Slot, extract_slots
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["SecondaryReport", "score_secondary_track", "apply_secondary_track"]


@dataclass(slots=True)
class SecondaryReport:
    score: float
    detach_events: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _primary_secondary(slots: list[Slot]) -> tuple[Slot | None, Slot | None]:
    if not slots:
        return None, None
    ordered = sorted(slots, key=lambda s: -s.mass)
    primary = ordered[0]
    secondary = None
    best = 1e9
    for s in ordered[1:6]:
        dist = ((s.cy - primary.cy) ** 2 + (s.cx - primary.cx) ** 2) ** 0.5
        # Prefer nearby mid-mass companions (props), not far background
        if 4.0 < dist < 80.0 and s.mass > 2.0 and dist < best:
            best = dist
            secondary = s
    return primary, secondary


def score_secondary_track(frame_paths: list[Path] | list[str], *, sample_every: int = 1) -> SecondaryReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return SecondaryReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    detach = 0
    dists: list[float] = []
    prev_sec: Slot | None = None
    h = w = 1
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=10)
        prim, sec = _primary_secondary(slots)
        if prim is None or sec is None:
            continue
        d = ((sec.cy - prim.cy) ** 2 + (sec.cx - prim.cx) ** 2) ** 0.5
        dists.append(d)
        if prev_sec is not None:
            # Secondary jumped far from previous secondary without primary moving much
            jump = ((sec.cy - prev_sec.cy) ** 2 + (sec.cx - prev_sec.cx) ** 2) ** 0.5
            if jump > max(h, w) * 0.2:
                detach += 1
        prev_sec = sec
    if not dists:
        return SecondaryReport(score=0.85, notes=["no_secondary_slot"])
    stab = float(np.clip(1.0 - float(np.std(dists)) / (max(h, w) * 0.15 + 1e-6), 0.0, 1.0))
    score = float(np.clip(stab - detach * 0.12, 0.0, 1.0))
    notes = [f"detach_events={detach}"] if detach else []
    return SecondaryReport(score=score, detach_events=detach, notes=notes)


def apply_secondary_track(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
) -> SecondaryReport:
    """Reinject last-seen secondary patch when it detaches from the hero."""
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return SecondaryReport(score=1.0)
    repaired = 0
    prev_rgb = read_frame_rgb(paths[0])
    prev_slots = extract_slots(prev_rgb, k=10)
    _, prev_sec = _primary_secondary(prev_slots)
    h, w = prev_rgb.shape[:2]
    a = float(np.clip(strength, 0.0, 1.0))
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        slots = extract_slots(curr, k=10)
        prim, sec = _primary_secondary(slots)
        if prev_sec is None:
            prev_rgb = curr
            prev_slots = slots
            prev_sec = sec
            continue
        lost = False
        if sec is None and prim is not None:
            lost = True
        elif sec is not None and prev_sec is not None:
            jump = ((sec.cy - prev_sec.cy) ** 2 + (sec.cx - prev_sec.cx) ** 2) ** 0.5
            if jump > max(h, w) * 0.2:
                lost = True
        if lost and prev_sec is not None:
            y0, x0, y1, x1 = prev_sec.bbox
            # Offset bbox toward current primary if available
            if prim is not None:
                dy = int(prim.cy - (prev_slots[0].cy if prev_slots else prim.cy))
                dx = int(prim.cx - (prev_slots[0].cx if prev_slots else prim.cx))
                y0, y1 = max(0, y0 + dy), min(h, y1 + dy)
                x0, x1 = max(0, x0 + dx), min(w, x1 + dx)
            patch = prev_rgb[prev_sec.bbox[0] : prev_sec.bbox[2], prev_sec.bbox[1] : prev_sec.bbox[3]]
            th, tw = y1 - y0, x1 - x0
            if patch.size and th > 0 and tw > 0:
                from PIL import Image

                patch_r = np.asarray(Image.fromarray(patch).resize((tw, th), Image.BILINEAR))
                region = curr[y0:y1, x0:x1].astype(np.float32)
                curr[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch_r.astype(np.float32) * a, 0, 255).astype(
                    np.uint8
                )
                save_frame_rgb(p, curr)
                repaired += 1
        prev_rgb = curr
        prev_slots = extract_slots(curr, k=10)
        _, prev_sec = _primary_secondary(prev_slots)
    after = score_secondary_track(paths)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"secondary_repaired={repaired}")
    return after
