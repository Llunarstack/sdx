"""Shared manifest helpers for scrape -> enrich -> train and the JSONL CLI tools."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_manifest_rows(path: Path) -> Iterator[dict]:
    """Stream JSONL rows, skipping blank and malformed lines."""
    with Path(path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_manifest_rows(path: Path) -> list[dict]:
    return list(iter_manifest_rows(path)) if path.is_file() else []


def first_str(row: dict, keys: Iterable[str]) -> str:
    """First non-empty string value of ``row`` among ``keys`` (stripped), else ``""``."""
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def caption_from_row(row: dict) -> str:
    return first_str(row, ("caption", "text", "prompt"))


def resolve_image_path(rel: str, *roots: Path | None) -> Path | None:
    """``rel`` itself if it is a file, else the first ``root / rel`` that is; ``None`` when nothing exists."""
    if not rel:
        return None
    for cand in (Path(rel), *(root / rel for root in roots if root is not None)):
        if cand.is_file():
            return cand
    return None


def manifest_has_rows(path: Path) -> bool:
    """True when ``path`` is a file with at least one parseable row (stops at the first)."""
    return path.is_file() and any(True for _ in iter_manifest_rows(path))


def pick_training_manifest(combined: Path, enriched: Path) -> Path:
    """Prefer enriched manifest when it has rows (avoids empty partial writes)."""
    if manifest_has_rows(enriched):
        return enriched
    if combined.is_file():
        return combined
    raise FileNotFoundError(f"no manifest (combined={combined}, enriched={enriched})")


def negative_caption_from_row(row: dict) -> str:
    """Map enrich / research fields to the training negative caption."""
    return first_str(row, ("negative_caption", "negative_prompt", "negative_prompt_hint"))


_SCREENED_FIELDS = (
    "caption",
    "text",
    "caption_global",
    "caption_local",
    "entity_captions",
    "region_captions",
    "segments",
    "parts",
    "character_tags",
    "copyright_tags",
    "wd_tags",
    "prompt_layout",
)


def _field_text(val, *, space_separated: bool = False) -> str:
    if isinstance(val, str):
        return ", ".join(val.split()) if space_separated else val
    if isinstance(val, dict):
        val = list(val.values())
    if isinstance(val, (list, tuple)):
        return ", ".join(t for t in (_field_text(v) for v in val) if t)
    return ""


def manifest_row_text(row: dict) -> str:
    """Every caption, tag and layout field of a manifest row, for training-data safety screening.

    Booru ``tag_string*`` fields (and a comma-free ``tags`` string) are space-separated tags,
    so they are re-joined with commas.
    """
    parts = [_field_text(row.get(k)) for k in _SCREENED_FIELDS]
    tags = row.get("tags")
    parts.append(_field_text(tags, space_separated=isinstance(tags, str) and "," not in tags))
    parts += [_field_text(v, space_separated=True) for k, v in row.items() if k.startswith("tag_string")]
    return ", ".join(p for p in parts if p)
