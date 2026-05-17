"""
Helpers for ``book_manifest.json`` produced by ``generate_book.py --write-book-manifest``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pipelines.book_comic.book_helpers import pick_metric_requires_vit_ckpt


def load_book_manifest(path: Union[str, Path]) -> Dict[str, Any]:
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


def manifest_project_root(manifest_path: Union[str, Path]) -> Path:
    """Directory containing manifest assets (parent of the JSON file)."""
    p = Path(manifest_path)
    if p.is_dir():
        return p.resolve()
    return p.parent.resolve()


def manifest_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = manifest.get("entries")
    if not isinstance(raw, list):
        return []
    return [e for e in raw if isinstance(e, dict)]


def manifest_page_indices(entries: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for e in entries:
        if e.get("kind") != "page":
            continue
        if "index" in e:
            try:
                out.append(int(e["index"]))
            except (TypeError, ValueError):
                pass
    return sorted(set(out))


def manifest_prompt_digest(
    entries: List[Dict[str, Any]],
    *,
    max_chars: int = 100,
    kind: str = "page",
) -> List[Tuple[int, str]]:
    """Return (page_index, truncated prompt) for manifest rows."""
    rows: List[Tuple[int, str]] = []
    cover_kinds = frozenset({"cover", "front_cover", "back_cover"})
    for e in entries:
        ek = e.get("kind")
        if kind == "cover" and ek not in cover_kinds:
            continue
        if kind != "cover" and ek != kind:
            continue
        try:
            idx = int(e.get("index", -1))
        except (TypeError, ValueError):
            idx = -1
        pr = str(e.get("prompt", "") or "").strip()
        if max_chars > 0 and len(pr) > max_chars:
            pr = pr[: max_chars - 1] + "…"
        rows.append((idx, pr))
    rows.sort(key=lambda t: t[0])
    return rows


def manifest_summary_lines(manifest: Dict[str, Any]) -> List[str]:
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
    sc = manifest.get("sample_candidates")
    if sc is not None:
        try:
            lines.append(f"sample_candidates={int(sc)}")
        except (TypeError, ValueError):
            pass
    pv = manifest.get("pick_vit_ckpt")
    if pv:
        lines.append("pick_vit_ckpt=yes")
    vm = manifest.get("visual_memory")
    if vm:
        lines.append(f"visual_memory={vm}")
    ids = manifest.get("visual_memory_entity_ids")
    if isinstance(ids, list) and ids:
        lines.append(f"visual_memory_entities={len(ids)}")
    ck = manifest.get("consistency_json")
    if ck:
        lines.append("consistency_json=yes")
    return lines


def validate_book_manifest(
    manifest: Dict[str, Any],
    *,
    project_root: Union[str, Path, None] = None,
    check_files: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Sanity-check a ``book_manifest.json`` dict (post-run QC).

    Returns ``(errors, warnings)``. Errors are structural problems; warnings flag weak configs.
    """
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(manifest, dict):
        return (["manifest root must be an object"], [])
    ckpt = str(manifest.get("ckpt") or "").strip()
    if not ckpt:
        errors.append("missing ckpt")
    entries = manifest_entries(manifest)
    if not entries:
        warnings.append("entries list is empty")
    seen_idx: Dict[int, int] = {}
    for e in entries:
        if e.get("kind") != "page":
            continue
        try:
            idx = int(e.get("index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0:
            continue
        seen_idx[idx] = seen_idx.get(idx, 0) + 1
    for idx, n in seen_idx.items():
        if n > 1:
            warnings.append(f"duplicate manifest page index {idx} ({n} rows)")

    pb = str(manifest.get("pick_best") or "").strip().lower()
    pv = manifest.get("pick_vit_ckpt")
    pv_s = str(pv).strip() if pv is not None else ""
    if pick_metric_requires_vit_ckpt(pb) and not pv_s:
        warnings.append(f"pick_best={pb!r} but pick_vit_ckpt missing in manifest (run may have used neutral ViT scores)")

    bw = manifest.get("beam_width")
    try:
        bw_i = int(bw or 0)
    except (TypeError, ValueError):
        bw_i = 0
    sc = manifest.get("sample_candidates")
    try:
        sc_i = int(sc or 0)
    except (TypeError, ValueError):
        sc_i = 0
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
