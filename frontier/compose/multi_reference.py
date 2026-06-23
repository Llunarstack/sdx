"""
Per-region reference images (Regional-Prompting-FLUX + PULID pattern).

Attach identity/style refs to individual boxes before sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RegionReference:
    region_name: str
    path: Path
    weight: float = 0.8
    mode: str = "identity"  # identity | style | structure


def _resolve(path: str, source_dir: Optional[Path]) -> Optional[Path]:
    raw = (path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file() and source_dir is not None:
        p = source_dir / raw
    return p if p.is_file() else None


def references_from_layout(
    regions: Sequence[Any],
    *,
    source_dir: Optional[Path] = None,
) -> List[RegionReference]:
    """Collect ``reference`` / ``reference_image`` paths from layout regions."""
    out: List[RegionReference] = []
    for r in regions:
        ref = str(
            getattr(r, "reference_path", "")
            or getattr(r, "reference", "")
            or ""
        ).strip()
        if not ref:
            continue
        p = _resolve(ref, source_dir)
        if p is None:
            continue
        w = float(getattr(r, "reference_weight", 0.8) or 0.8)
        mode = str(getattr(r, "reference_mode", "identity") or "identity")
        out.append(
            RegionReference(
                region_name=str(getattr(r, "name", "region")),
                path=p,
                weight=max(0.0, min(1.0, w)),
                mode=mode,
            )
        )
    return out


def merge_reference_prompts(prompt: str, refs: Sequence[RegionReference]) -> str:
    """Light prompt augmentation when reference images are attached."""
    p = (prompt or "").strip()
    if not refs:
        return p
    tags = []
    for ref in refs:
        if ref.mode == "identity":
            tags.append(f"match identity from reference for {ref.region_name}")
        elif ref.mode == "style":
            tags.append(f"match style from reference for {ref.region_name}")
        else:
            tags.append(f"match structure from reference for {ref.region_name}")
    suffix = ", ".join(tags)
    return f"{p}, {suffix}" if p else suffix
