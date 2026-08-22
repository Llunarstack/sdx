"""Shared manifest path resolution for scrape → enrich → train."""

from __future__ import annotations

import json
from pathlib import Path


def read_manifest_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def manifest_has_rows(path: Path) -> bool:
    return bool(read_manifest_rows(path))


def pick_training_manifest(combined: Path, enriched: Path) -> Path:
    """Prefer enriched manifest when it has rows (avoids empty partial writes)."""
    if manifest_has_rows(enriched):
        return enriched
    if combined.is_file():
        return combined
    raise FileNotFoundError(f"no manifest (combined={combined}, enriched={enriched})")


def negative_caption_from_row(row: dict) -> str:
    """Map enrich / research fields to the training negative caption."""
    for key in ("negative_caption", "negative_prompt", "negative_prompt_hint"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
