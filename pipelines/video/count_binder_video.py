"""
Video count binder — secondary objects that vanish mid-clip (prompt says "three").

Closed APIs drop people/props when scenes get busy. We track blob-count stability
and reinject missing mass from the frame that matched the expected count best.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permanence import extract_slots
from .video_io import read_frame_rgb, save_frame_rgb

__all__ = ["CountBindVideoReport", "score_count_stability", "apply_count_bind"]


@dataclass(slots=True)
class CountBindVideoReport:
    score: float
    expected: int = 0
    mean_count: float = 0.0
    drift_frames: int = 0
    repaired: int = 0
    notes: list[str] = field(default_factory=list)


def _blob_count(rgb: np.ndarray, *, k: int = 12) -> int:
    slots = extract_slots(rgb, k=k)
    return sum(1 for s in slots if s.mass > 4.0)


def expected_count_from_prompt(prompt: str) -> int:
    """Cheap numeral / word-count parse without heavy NLP."""
    text = (prompt or "").lower()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "a pair": 2,
        "couple": 2,
        "trio": 3,
    }
    for w, n in words.items():
        if w in text:
            return n
    import re

    m = re.search(r"\b([2-9]|1[0-2])\b", text)
    if m:
        return int(m.group(1))
    try:
        from utils.quality.count_binder_eval import analyze_prompt_counts

        rep = analyze_prompt_counts(prompt)
        if rep.expected_counts:
            return max(rep.expected_counts.values())
        return max(1, int(rep.expected_subjects))
    except Exception:
        return 0


def score_count_stability(
    frame_paths: list[Path] | list[str],
    *,
    expected: int = 0,
    prompt: str = "",
    sample_every: int = 2,
) -> CountBindVideoReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if not paths:
        return CountBindVideoReport(score=1.0)
    exp = int(expected) if expected > 0 else expected_count_from_prompt(prompt)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    counts = [_blob_count(read_frame_rgb(paths[i])) for i in idxs]
    mean_c = float(np.mean(counts)) if counts else 0.0
    if exp <= 0:
        # Stability-only: variance of count
        var = float(np.var(counts)) if len(counts) > 1 else 0.0
        score = float(np.clip(1.0 - var * 0.35, 0.0, 1.0))
        return CountBindVideoReport(score=score, expected=0, mean_count=mean_c, notes=["count_stability_only"])
    drift = sum(1 for c in counts if abs(c - exp) > 1)
    err = float(np.mean([abs(c - exp) for c in counts]))
    score = float(np.clip(1.0 - err / max(exp, 1) - drift * 0.05, 0.0, 1.0))
    notes = []
    if drift:
        notes.append(f"count_drift_frames={drift}")
    return CountBindVideoReport(score=score, expected=exp, mean_count=mean_c, drift_frames=drift, notes=notes)


def apply_count_bind(
    frame_paths: list[Path],
    *,
    expected: int = 0,
    prompt: str = "",
    strength: float = 0.4,
) -> CountBindVideoReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return CountBindVideoReport(score=1.0)
    exp = int(expected) if expected > 0 else expected_count_from_prompt(prompt)
    counts = [_blob_count(read_frame_rgb(p)) for p in paths]
    if exp <= 0:
        exp = int(round(float(np.median(counts)))) if counts else 0
    # Donor = frame closest to expected count
    donor_i = int(np.argmin([abs(c - exp) for c in counts])) if counts else 0
    donor = read_frame_rgb(paths[donor_i])
    donor_slots = [s for s in extract_slots(donor, k=12) if s.mass > 4.0]
    repaired = 0
    a = float(np.clip(strength, 0.0, 1.0))
    for i, p in enumerate(paths):
        if abs(counts[i] - exp) <= 1:
            continue
        curr = read_frame_rgb(p)
        # Paste missing donor slots that aren't near any current slot
        curr_slots = extract_slots(curr, k=12)
        for ds in donor_slots:
            near = any(
                ((ds.cy - cs.cy) ** 2 + (ds.cx - cs.cx) ** 2) ** 0.5 < 12.0 for cs in curr_slots if cs.mass > 3.0
            )
            if near:
                continue
            y0, x0, y1, x1 = ds.bbox
            patch = donor[y0:y1, x0:x1]
            region = curr[y0:y1, x0:x1].astype(np.float32)
            if patch.size == 0 or region.size == 0:
                continue
            curr[y0:y1, x0:x1] = np.clip(region * (1.0 - a) + patch.astype(np.float32) * a, 0, 255).astype(np.uint8)
            repaired += 1
            break  # one reinject per frame to avoid soup
        if repaired:
            save_frame_rgb(p, curr)
    after = score_count_stability(paths, expected=exp, sample_every=1)
    after.repaired = repaired
    if repaired:
        after.notes.append(f"count_repaired={repaired}")
    return after
