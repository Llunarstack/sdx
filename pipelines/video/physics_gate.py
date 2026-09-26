"""
Physics gate — Physical Invariance Score lite (NewtonGen / CausalMotion inspired).

SOTA models still invent impossible motion: objects accelerate upward "falling",
velocity reverses without braking, contacts ignore floors. We don't run a full
sim — we score trajectory sanity on permanence slots and fail/retry when broken.

Training-free, detector-light, actionable for segment_retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permanence import extract_slots, match_slots
from .video_io import read_frame_rgb

__all__ = [
    "PhysicsReport",
    "score_physics_invariance",
]


@dataclass(slots=True)
class PhysicsReport:
    """Higher = more physically plausible."""

    score: float
    velocity_flip_penalties: int = 0
    anti_gravity_events: int = 0
    jump_discontinuities: int = 0
    notes: list[str] = field(default_factory=list)


def score_physics_invariance(
    frame_paths: list[Path] | list[str],
    *,
    sample_every: int = 1,
    k_slots: int = 6,
) -> PhysicsReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 3:
        return PhysicsReport(score=1.0)

    idxs = list(range(0, len(paths), max(1, sample_every)))
    if idxs[-1] != len(paths) - 1:
        idxs.append(len(paths) - 1)

    # Track primary slot (highest mass) trajectories via greedy match
    trajectories: dict[int, list[tuple[float, float]]] = {}  # track_id -> [(cy,cx), ...]
    next_id = 0
    prev_slots = None
    prev_ids: list[int] = []
    h = w = 1

    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is None:
            prev_ids = list(range(len(slots)))
            next_id = len(slots)
            for tid, s in zip(prev_ids, slots):
                trajectories.setdefault(tid, []).append((s.cy, s.cx))
            prev_slots = slots
            continue
        pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.22)
        curr_ids = [-1] * len(slots)
        for pi, cj in pairs:
            tid = prev_ids[pi]
            curr_ids[cj] = tid
            trajectories.setdefault(tid, []).append((slots[cj].cy, slots[cj].cx))
        for cj in app:
            tid = next_id
            next_id += 1
            curr_ids[cj] = tid
            trajectories.setdefault(tid, []).append((slots[cj].cy, slots[cj].cx))
        prev_slots = slots
        prev_ids = curr_ids

    flips = anti_g = jumps = 0
    appear_pop = 0
    for pts in trajectories.values():
        if len(pts) < 3:
            continue
        vys = [pts[i][0] - pts[i - 1][0] for i in range(1, len(pts))]
        vxs = [pts[i][1] - pts[i - 1][1] for i in range(1, len(pts))]
        for i in range(1, len(vys)):
            # Abrupt direction reverse without near-zero speed
            speed_prev = (vys[i - 1] ** 2 + vxs[i - 1] ** 2) ** 0.5
            speed = (vys[i] ** 2 + vxs[i] ** 2) ** 0.5
            if speed_prev > 1.5 and speed > 1.5:
                # Dot product of consecutive velocities
                dot = vys[i - 1] * vys[i] + vxs[i - 1] * vxs[i]
                if dot < -0.5 * speed_prev * speed:
                    flips += 1
            # Anti-gravity: strong upward acceleration (cy decreases = up in image coords)
            ay = vys[i] - vys[i - 1]
            if ay < -3.0 and vys[i] < -2.0:  # accelerating upward hard
                anti_g += 1
            # Teleport discontinuity
            if speed > max(h, w) * 0.12:
                jumps += 1

    # Re-scan for pop teleports: massy slot disappears and another appears far away same step
    prev_slots = None
    for ii in idxs:
        rgb = read_frame_rgb(paths[ii])
        h, w = rgb.shape[:2]
        slots = extract_slots(rgb, k=k_slots)
        if prev_slots is not None:
            _pairs, dis, app = match_slots(prev_slots, slots, h=h, w=w, max_dist=0.22)
            massy_dis = [prev_slots[i] for i in dis if prev_slots[i].mass > 3.0]
            massy_app = [slots[j] for j in app if slots[j].mass > 3.0]
            if massy_dis and massy_app:
                # If a disappeared and appeared slot are far apart → teleport/pop
                for a in massy_dis:
                    for b in massy_app:
                        dist = ((a.cy - b.cy) ** 2 + (a.cx - b.cx) ** 2) ** 0.5
                        if dist > max(h, w) * 0.25:
                            appear_pop += 1
                            jumps += 1
        prev_slots = slots

    # Normalize penalties by trajectory-steps
    steps = max(1, sum(max(0, len(p) - 2) for p in trajectories.values()) + appear_pop)
    rate = (flips * 1.2 + anti_g * 1.5 + jumps * 1.0) / steps
    score = float(np.clip(1.0 - rate * 3.0, 0.0, 1.0))
    notes = []
    if flips:
        notes.append(f"velocity_flips={flips}")
    if anti_g:
        notes.append(f"anti_gravity={anti_g}")
    if jumps:
        notes.append(f"teleports={jumps}")
    if appear_pop:
        notes.append(f"pop_teleports={appear_pop}")
    return PhysicsReport(
        score=score,
        velocity_flip_penalties=flips,
        anti_gravity_events=anti_g,
        jump_discontinuities=jumps,
        notes=notes,
    )
