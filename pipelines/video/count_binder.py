"""
Count binder — secondary-object / person count continuity across a clip.

Competitors drop extras mid-shot ("three people" becomes two). We estimate
blob counts cheaply and fail/retry when the count collapses without a cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .permanence import extract_slots
from .video_io import read_frame_rgb

__all__ = ["CountReport", "score_count_stability", "expected_count_from_prompt"]


@dataclass(slots=True)
class CountReport:
    score: float
    expected: int = 0
    median_count: float = 0.0
    drop_events: int = 0
    notes: list[str] = field(default_factory=list)


def expected_count_from_prompt(prompt: str) -> int:
    """Best-effort expected subject count from prompt text."""
    try:
        from utils.quality.count_binder_eval import analyze_prompt_counts

        rep = analyze_prompt_counts(prompt or "")
        if rep.expected_counts:
            return max(int(v) for v in rep.expected_counts.values())
        return max(1, int(rep.expected_subjects))
    except Exception:
        text = (prompt or "").lower()
        for word, n in (
            ("five ", 5),
            ("four ", 4),
            ("three ", 3),
            ("two ", 2),
            ("a pair", 2),
            ("couple", 2),
            ("group of", 3),
        ):
            if word in text:
                return n
        return 1


def _massy_slot_count(rgb: np.ndarray, *, mass_min: float = 3.5) -> int:
    slots = extract_slots(rgb, k=12)
    return sum(1 for s in slots if s.mass >= mass_min)


def score_count_stability(
    frame_paths: list[Path] | list[str],
    *,
    expected: int = 0,
    prompt: str = "",
    sample_every: int = 2,
    drop_tol: float = 0.45,
) -> CountReport:
    paths = [Path(p) for p in frame_paths if Path(p).is_file()]
    if len(paths) < 2:
        return CountReport(score=1.0, expected=expected)
    exp = int(expected) if expected > 0 else expected_count_from_prompt(prompt)
    idxs = list(range(0, len(paths), max(1, sample_every)))
    counts = [_massy_slot_count(read_frame_rgb(paths[i])) for i in idxs]
    med = float(np.median(counts)) if counts else 0.0
    drops = 0
    for i in range(1, len(counts)):
        if counts[i - 1] >= 2 and counts[i] <= counts[i - 1] * (1.0 - drop_tol):
            drops += 1
    # Score vs expected + temporal stability
    if exp > 0 and med > 0:
        ratio = min(med, exp) / max(med, exp)
    else:
        ratio = 1.0
    stab = float(np.clip(1.0 - drops * 0.2, 0.0, 1.0))
    score = float(np.clip(0.55 * ratio + 0.45 * stab, 0.0, 1.0))
    notes = []
    if drops:
        notes.append(f"count_drops={drops}")
    if exp > 0 and abs(med - exp) >= 1.5:
        notes.append(f"count_median={med:.1f}_expected={exp}")
    return CountReport(score=score, expected=exp, median_count=med, drop_events=drops, notes=notes)
