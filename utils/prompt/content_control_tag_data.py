"""
Load content-control tag packs from ``data/prompt_tags/*.csv``.

Each CSV uses the same columns::

    pack,mode,tag

- ``pack`` — logical name (e.g. ``pose_positive``, ``sfw_positive``).
- ``mode`` — sub-key for dict-valued packs; use ``_`` for flat lists (single mode).
- ``tag`` — prompt token, or a row with ``tag`` = ``__REF__:other_pack`` to splice
  all ``_*`` tags from ``other_pack`` (flat ``_`` mode) in order.

Rows are read in file order; UTF-8 with BOM allowed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_TAG_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "prompt_tags"


def _expand_refs_in_table(raw: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    """Expand ``__REF__:pack`` rows by splicing the referenced pack's flat ``_`` tags (in order)."""

    def expand_list(tags: List[str], stack: List[str]) -> List[str]:
        out: List[str] = []
        for t in tags:
            if t.startswith("__REF__:"):
                name = t[8:].strip()
                if name in stack:
                    raise ValueError(f"circular __REF__: {' -> '.join(stack + [name])}")
                ref_flat = raw.get(name, {}).get("_", [])
                out.extend(expand_list(ref_flat, stack + [name]))
            else:
                out.append(t)
        return out

    return {p: {m: expand_list(tags, []) for m, tags in modes.items()} for p, modes in raw.items()}


def load_tag_tables(
    directory: Path | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Merge all ``*.csv`` under ``directory`` into ``{pack: {mode: [tags...]}}``.
    Missing directory or files returns empty dict (caller may fall back).
    """
    root = directory if directory is not None else _TAG_DATA_DIR
    if not root.is_dir():
        return {}

    merged: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "pack" not in reader.fieldnames:
                    continue
                for row in reader:
                    pack = (row.get("pack") or "").strip()
                    if not pack or pack.startswith("#"):
                        continue
                    mode = (row.get("mode") or "_").strip() or "_"
                    tag = (row.get("tag") or "").strip()
                    if not tag or tag.startswith("#"):
                        continue
                    merged[pack][mode].append(tag)
        except OSError:
            continue

    # defaultdict -> plain dict
    plain = {pk: dict(modes) for pk, modes in merged.items()}
    return _expand_refs_in_table(plain)


def flat_pack(tables: Dict[str, Dict[str, List[str]]], pack: str) -> List[str]:
    return list((tables.get(pack) or {}).get("_", []))


def dict_pack(tables: Dict[str, Dict[str, List[str]]], pack: str) -> Dict[str, List[str]]:
    d = tables.get(pack) or {}
    return {k: list(v) for k, v in d.items()}


def conflicting_pairs_from_table(tables: Dict[str, Dict[str, List[str]]]) -> List[Tuple[str, str]]:
    rows = flat_pack(tables, "conflicting_tag_pairs")
    out: List[Tuple[str, str]] = []
    for r in rows:
        if "|||" in r:
            a, b = r.split("|||", 1)
            out.append((a.strip(), b.strip()))
    return out


def default_tag_data_dir() -> Path:
    return _TAG_DATA_DIR
