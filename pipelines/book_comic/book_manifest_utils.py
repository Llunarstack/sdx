"""
Helpers for ``book_manifest.json`` produced by ``generate_book.py --write-book-manifest``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipelines.book_comic.book_helpers import pick_metric_requires_vit_ckpt


def load_book_manifest(path: str | Path) -> dict[str, Any]:
    """Load ``book_manifest.json`` or ``book.json`` from a project directory or file path."""
    p = Path(path)
    if p.is_dir():
        for name in ("book_manifest.json", "book.json"):
            candidate = p / name
            if candidate.is_file():
                p = candidate
                break
        else:
            raise FileNotFoundError(f"No book_manifest.json or book.json under {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("book manifest root must be an object")
    return data


def manifest_project_root(manifest_path: str | Path) -> Path:
    """Directory containing manifest assets (parent of the JSON file)."""
    p = Path(manifest_path)
    return (p if p.is_dir() else p.parent).resolve()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Dict rows of ``manifest["entries"]`` (non-dict rows are dropped)."""
    raw = manifest.get("entries")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


def manifest_summary_lines(manifest: dict[str, Any]) -> list[str]:
    """Human-readable one-liners for logs or CI artifacts."""
    ent = manifest_entries(manifest)
    pages = [e for e in ent if e.get("kind") == "page"]
    covers = [e for e in ent if e.get("kind") in ("cover", "front_cover")]
    back_covers = [e for e in ent if e.get("kind") == "back_cover"]
    lines = [
        f"pages_recorded={len(pages)}",
        f"front_covers_recorded={len(covers)}",
        f"back_covers_recorded={len(back_covers)}",
    ]
    pb = str(manifest.get("pick_best") or "").strip()
    if pb:
        lines.append(f"pick_best={pb}")
    sc = _int_or_none(manifest.get("sample_candidates"))
    if sc is not None:
        lines.append(f"sample_candidates={sc}")
    if manifest.get("pick_vit_ckpt"):
        lines.append("pick_vit_ckpt=yes")
    vm = manifest.get("visual_memory")
    if vm:
        lines.append(f"visual_memory={vm}")
    ids = manifest.get("visual_memory_entity_ids")
    if isinstance(ids, list) and ids:
        lines.append(f"visual_memory_entities={len(ids)}")
    if manifest.get("consistency_json"):
        lines.append("consistency_json=yes")
    return lines


def validate_book_manifest(
    manifest: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    check_files: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Sanity-check a ``book_manifest.json`` dict (post-run QC).

    Returns ``(errors, warnings)``. Errors are structural problems; warnings flag weak configs.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(manifest, dict):
        return (["manifest root must be an object"], [])
    ckpt = str(manifest.get("ckpt") or "").strip()
    if not ckpt:
        errors.append("missing ckpt")
    entries = manifest_entries(manifest)
    if not entries:
        warnings.append("entries list is empty")
    page_indices = Counter(
        idx
        for e in entries
        if e.get("kind") == "page" and (idx := _int_or_none(e.get("index"))) is not None and idx >= 0
    )
    for idx, n in page_indices.items():
        if n > 1:
            warnings.append(f"duplicate manifest page index {idx} ({n} rows)")

    pb = str(manifest.get("pick_best") or "").strip().lower()
    pv = manifest.get("pick_vit_ckpt")
    pv_s = str(pv).strip() if pv is not None else ""
    if pick_metric_requires_vit_ckpt(pb) and not pv_s:
        warnings.append(
            f"pick_best={pb!r} but pick_vit_ckpt missing in manifest (run may have used neutral ViT scores)"
        )

    bw_i = _int_or_none(manifest.get("beam_width")) or 0
    sc_i = _int_or_none(manifest.get("sample_candidates")) or 0
    if bw_i > 1 and sc_i > 1:
        warnings.append(f"beam_width={bw_i} with sample_candidates={sc_i} (beam is meant for num=1 branch)")

    kinds = {str(e.get("kind") or "") for e in entries}
    if "front_cover" not in kinds and "cover" not in kinds:
        warnings.append("no front cover entry (kind front_cover or legacy cover)")
    if "back_cover" not in kinds and any(k in kinds for k in ("front_cover", "cover", "page")):
        warnings.append("no back_cover entry (optional but recommended for print wrap)")

    if check_files and project_root is not None:
        root = Path(project_root).resolve()
        for e in entries:
            rel = str(e.get("path") or "").strip()
            if not rel:
                warnings.append(f"entry kind={e.get('kind')!r} missing path")
                continue
            fp = (root / Path(rel.replace("\\", "/"))).resolve()
            if not fp.is_file():
                errors.append(f"missing file: {rel}")

    return errors, warnings
