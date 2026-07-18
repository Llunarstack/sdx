"""Parallel segment processing for multi-core / multi-GPU pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["SegmentWorkItem", "SegmentWorkResult", "run_segments_parallel"]


@dataclass(slots=True)
class SegmentWorkItem:
    index: int
    seg_dir: Path


@dataclass(slots=True)
class SegmentWorkResult:
    index: int
    frames: list[Path]
    quality: Any


def run_segments_parallel(
    items: Sequence[SegmentWorkItem],
    worker: Callable[[SegmentWorkItem], tuple[list[Path], Any]],
    *,
    max_workers: int = 2,
) -> list[SegmentWorkResult]:
    """Run independent segment jobs; results sorted by segment index."""
    if max_workers <= 1 or len(items) <= 1:
        out = [SegmentWorkResult(i.index, *worker(i)) for i in items]
        return sorted(out, key=lambda r: r.index)

    results: list[SegmentWorkResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futs = {pool.submit(worker, item): item for item in items}
        for fut in as_completed(futs):
            item = futs[fut]
            frames, quality = fut.result()
            results.append(SegmentWorkResult(item.index, frames, quality))
    return sorted(results, key=lambda r: r.index)
