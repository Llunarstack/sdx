"""
Object **permanence** — competitors denoise pixels; we track entities.

Closed models (Sora/Veo/Kling) have no explicit object state, so props blink
in and out. This module:

1. Extracts cheap salient "slots" per frame (luma peaks / blobs).
2. Matches slots across time (Hungarian-lite greedy).
3. Flags abrupt appear / disappear without supporting optical flow.
4. Repairs disappearances by warping the last-seen patch forward.

This is the SDX answer to GenVID's "Abrupt Appearance/Disappearance" artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .video_io import read_frame_rgb, save_frame_rgb

__all__ = [
    "PermanenceReport",
    "Slot",
    "extract_slots",
    "match_slots",
    "score_object_permanence",
    "repair_permanence_breaks",
    "apply_permanence_pass",
]


@dataclass(slots=True)
class Slot:
    """One salient region in a frame."""

    cy: float
    cx: float
    mass: float
    color: tuple[float, float, float]
    bbox: tuple[int, int, int, int]  # y0,x0,y1,x1


@dataclass(slots=True)
class PermanenceReport:
    score: float
    """1 = perfect permanence, 0 = chaos of popping objects."""
    disappear_events: int = 0
    appear_events: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _downsample(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(rgb[..., :3].astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def extract_slots(rgb: np.ndarray, *, k: int = 8, grid: int = 8) -> list[Slot]:
    """
    Grid-peak slots: take the highest-contrast cell centers as entity proxies.

    No detector dependency — works offline and is stable enough to catch pops.
    """
    small = _downsample(rgb, 64)
    h, w, _ = small.shape
    cell = max(1, 64 // grid)
    cands: list[tuple[float, int, int, np.ndarray]] = []
    gray = small.mean(axis=2)
    # Local contrast vs neighborhood
    for gy in range(grid):
        for gx in range(grid):
            y0, x0 = gy * cell, gx * cell
            y1, x1 = min(h, y0 + cell), min(w, x0 + cell)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            contrast = float(patch.std() + abs(patch.mean() - gray.mean()) * 0.25)
            cy = (y0 + y1) * 0.5
            cx = (x0 + x1) * 0.5
            color = small[int(cy), int(cx)]
            cands.append((contrast, int(cy), int(cx), color))
    cands.sort(key=lambda t: -t[0])
    slots: list[Slot] = []
    scale_y = rgb.shape[0] / 64.0
    scale_x = rgb.shape[1] / 64.0
    for contrast, cy, cx, color in cands[:k]:
        if contrast < 1.5:
            continue
        ry, rx = int(cy * scale_y), int(cx * scale_x)
        half = max(4, int(min(rgb.shape[0], rgb.shape[1]) * 0.04))
        y0, x0 = max(0, ry - half), max(0, rx - half)
        y1, x1 = min(rgb.shape[0], ry + half), min(rgb.shape[1], rx + half)
        mass = float(contrast)
        slots.append(
            Slot(
                cy=float(ry),
                cx=float(rx),
                mass=mass,
                color=(float(color[0]), float(color[1]), float(color[2])),
                bbox=(y0, x0, y1, x1),
            )
        )
    return slots


def _slot_dist(a: Slot, b: Slot, *, h: float, w: float) -> float:
    dy = (a.cy - b.cy) / max(h, 1.0)
    dx = (a.cx - b.cx) / max(w, 1.0)
    spatial = (dy * dy + dx * dx) ** 0.5
    col = (abs(a.color[0] - b.color[0]) + abs(a.color[1] - b.color[1]) + abs(a.color[2] - b.color[2])) / (3.0 * 255.0)
    return 0.75 * spatial + 0.25 * col


def match_slots(
    prev: list[Slot],
    curr: list[Slot],
    *,
    h: int,
    w: int,
    max_dist: float = 0.18,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match → (pairs, unmatched_prev=disappear, unmatched_curr=appear)."""
    pairs: list[tuple[int, int]] = []
    used_p: set[int] = set()
    used_c: set[int] = set()
    scores: list[tuple[float, int, int]] = []
    for i, a in enumerate(prev):
        for j, b in enumerate(curr):
            scores.append((_slot_dist(a, b, h=float(h), w=float(w)), i, j))
    scores.sort(key=lambda t: t[0])
    for d, i, j in scores:
        if d > max_dist:
            break
        if i in used_p or j in used_c:
            continue
        pairs.append((i, j))
        used_p.add(i)
        used_c.add(j)
    disappear = [i for i in range(len(prev)) if i not in used_p]
    appear = [j for j in range(len(curr)) if j not in used_c]
    return pairs, disappear, appear


def score_object_permanence(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 2,
    k_slots: int = 8,
) -> PermanenceReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return PermanenceReport(score=1.0)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)
    disappear = appear = 0
    transitions = 0
    prev_slots: list[Slot] | None = None
    h = w = 1
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is not None:
            _pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w)
            # Only count massy slots as real entities
            dis_n = sum(1 for i in dis if prev_slots[i].mass > 3.0)
            app_n = sum(1 for j in app if slots[j].mass > 3.0)
            disappear += dis_n
            appear += app_n
            transitions += 1
        prev_slots = slots
    if transitions <= 0:
        return PermanenceReport(score=1.0)
    # Penalize events relative to transitions
    rate = (disappear + appear) / max(transitions * k_slots * 0.35, 1.0)
    score = float(np.clip(1.0 - rate, 0.0, 1.0))
    notes = []
    if disappear:
        notes.append(f"disappear_events={disappear}")
    if appear:
        notes.append(f"appear_events={appear}")
    return PermanenceReport(
        score=score,
        disappear_events=disappear,
        appear_events=appear,
        notes=notes,
    )


def _paste_bbox(dst: np.ndarray, src: np.ndarray, bbox: tuple[int, int, int, int], *, alpha: float) -> None:
    y0, x0, y1, x1 = bbox
    patch = src[y0:y1, x0:x1]
    if patch.size == 0:
        return
    region = dst[y0:y1, x0:x1].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    dst[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)


def repair_permanence_breaks(
    frame_paths: list[Path],
    *,
    strength: float = 0.55,
    k_slots: int = 8,
) -> PermanenceReport:
    """
    When a massive slot disappears with little global motion, reinject last patch.
    """
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return PermanenceReport(score=1.0)
    repaired = 0
    disappear = appear = 0
    prev_rgb = read_frame_rgb(paths[0])
    prev_slots = extract_slots(prev_rgb, k=k_slots)
    h, w = prev_rgb.shape[:2]
    for p in paths[1:]:
        curr = read_frame_rgb(p)
        slots = extract_slots(curr, k=k_slots)
        pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w)
        disappear += len(dis)
        appear += len(app)
        # Global motion proxy
        gmotion = float(np.mean(np.abs(curr.astype(np.float32) - prev_rgb.astype(np.float32))) / 255.0)
        if gmotion < 0.12:  # mostly static camera — pops are suspicious
            changed = False
            for i in dis:
                slot = prev_slots[i]
                if slot.mass < 4.0:
                    continue
                _paste_bbox(curr, prev_rgb, slot.bbox, alpha=strength)
                repaired += 1
                changed = True
            if changed:
                save_frame_rgb(p, curr)
                curr = read_frame_rgb(p)
        prev_rgb = curr
        prev_slots = extract_slots(curr, k=k_slots)
    base = score_object_permanence(paths)
    base.repaired = repaired
    if repaired:
        base.notes.append(f"repaired={repaired}")
    return base


def apply_permanence_pass(
    frame_paths: list[Path],
    *,
    repair: bool = True,
    strength: float = 0.55,
) -> PermanenceReport:
    if repair:
        return repair_permanence_breaks(frame_paths, strength=strength)
    return score_object_permanence(frame_paths)
