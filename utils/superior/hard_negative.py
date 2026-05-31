"""
Mine **hard negative prompt tokens** from failed benchmark / hard-case exports.

Appends domain-specific negatives to reduce recurring failure modes (blur, text, counting).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

_TAG_NEGATIVES: Dict[str, str] = {
    "low_composite": "low quality, blurry, artifacts",
    "text_rendering": "misspelled text, illegible lettering, garbled typography",
    "counting": "wrong number of objects, extra limbs, duplicated subjects",
    "exposure": "overexposed, underexposed, blown highlights, crushed shadows",
    "oversaturation": "oversaturated, neon colors, unrealistic skin tones",
    "blur": "blurry, out of focus, soft details, motion blur",
}


@dataclass(slots=True)
class HardNegativeBundle:
    tags: List[str]
    negative_suffix: str
    per_tag_counts: Dict[str, int]


def tags_from_benchmark_row(row: Dict[str, Any], *, threshold: float = 0.60) -> List[str]:
    """Infer failure tags from a benchmark ``results.json`` row (mirrors benchmark_suite)."""
    tags: List[str] = []
    comp = float(row.get("composite", 0.0) or 0.0)
    if comp < threshold:
        tags.append("low_composite")
    if float(row.get("ocr_match", 1.0) or 1.0) < 0.7 and row.get("expected_text"):
        tags.append("text_rendering")
    if float(row.get("count_match", 1.0) or 1.0) < 0.7 and int(row.get("expected_count", 0) or 0) > 0:
        tags.append("counting")
    if float(row.get("exposure_balance", 0.5) or 0.5) < 0.6:
        tags.append("exposure")
    if float(row.get("saturation_balance", 0.5) or 0.5) < 0.6:
        tags.append("oversaturation")
    edge = float(row.get("edge_sharpness", 0.0) or 0.0)
    if edge / 400.0 < 0.35:
        tags.append("blur")
    return tags


def mine_hard_negatives(
    rows: Sequence[Dict[str, Any]],
    *,
    threshold: float = 0.60,
    max_phrases: int = 12,
) -> HardNegativeBundle:
    """Aggregate failure tags and build a negative prompt suffix."""
    counter: Counter[str] = Counter()
    for row in rows:
        for t in tags_from_benchmark_row(row, threshold=threshold):
            counter[t] += 1
        for t in row.get("failure_tags") or []:
            counter[str(t)] += 1
    ordered = [t for t, _ in counter.most_common(max_phrases)]
    parts: List[str] = []
    seen: set[str] = set()
    for tag in ordered:
        phrase = _TAG_NEGATIVES.get(tag, "")
        for chunk in phrase.split(","):
            c = chunk.strip().lower()
            if c and c not in seen:
                seen.add(c)
                parts.append(chunk.strip())
    suffix = ", ".join(parts)
    return HardNegativeBundle(tags=ordered, negative_suffix=suffix, per_tag_counts=dict(counter))


def load_hard_negatives_from_results(
    path: Union[str, Path],
    *,
    threshold: float = 0.60,
) -> HardNegativeBundle:
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = []
    low = [r for r in rows if isinstance(r, dict) and float(r.get("composite", 1.0) or 1.0) < threshold]
    return mine_hard_negatives(low or rows, threshold=threshold)


def merge_negative_prompt(base: str, bundle: HardNegativeBundle) -> str:
    """Append mined negative suffix to an existing negative prompt."""
    parts = [p.strip() for p in (base, bundle.negative_suffix) if p and p.strip()]
    return ", ".join(parts)


def benchmark_sample_args_for_negatives(bundle: HardNegativeBundle) -> List[str]:
    """Return ``benchmark_suite --sample-arg`` tokens for mined hard negatives."""
    if not bundle.negative_suffix.strip():
        return []
    return ["--negative-prompt", bundle.negative_suffix]


def load_and_mine_from_results(path: Union[str, Path], **kwargs: Any) -> HardNegativeBundle:
    """Alias for ``load_hard_negatives_from_results``."""
    return load_hard_negatives_from_results(path, **kwargs)


__all__ = [
    "HardNegativeBundle",
    "benchmark_sample_args_for_negatives",
    "load_and_mine_from_results",
    "load_hard_negatives_from_results",
    "merge_negative_prompt",
    "mine_hard_negatives",
    "tags_from_benchmark_row",
]
