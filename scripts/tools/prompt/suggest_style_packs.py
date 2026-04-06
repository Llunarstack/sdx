#!/usr/bin/env python3
"""
Suggest popular style presets from a free-text query.

Examples:
  python -m scripts.tools prompt_lint --help
  python -m scripts.tools suggest_style_packs --query "anime 3d game keyart" --limit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compact_cli_row(row: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {
        "id": str(row.get("id", "")),
        "category": str(row.get("category", "")),
        "score": str(row.get("score", "0")),
        "description": str(row.get("description", "")),
    }
    for k in (
        "lexicon_style",
        "art_medium_pack",
        "art_medium_family",
        "art_medium_variant",
        "artist_pack",
        "color_render_pack",
    ):
        v = str(row.get(k, "")).strip()
        if v:
            out[k] = v
    return out


def _format_as_command(row: Dict[str, str]) -> str:
    parts: List[str] = []
    if row.get("lexicon_style"):
        parts.extend(["--lexicon-style", row["lexicon_style"]])
    if row.get("art_medium_pack"):
        parts.extend(["--art-medium-pack", row["art_medium_pack"]])
    if row.get("art_medium_family"):
        parts.extend(["--art-medium-family", row["art_medium_family"]])
    if row.get("art_medium_variant"):
        parts.extend(["--art-medium-variant", row["art_medium_variant"]])
    if row.get("artist_pack"):
        parts.extend(["--artist-pack", row["artist_pack"]])
    if row.get("color_render_pack"):
        parts.extend(["--color-render-pack", row["color_render_pack"]])
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest style preset packs from free-text.")
    parser.add_argument("--query", type=str, default="", help="Free-text style query, e.g. 'anime 3d game keyart'.")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "digital", "3d", "drawing", "painting"],
        help="Optional category filter.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max rows to return.")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows.")
    args = parser.parse_args()

    import sys

    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from pipelines.book_comic.prompt_lexicon import suggest_popular_style_presets

    rows = suggest_popular_style_presets(
        str(args.query or ""),
        category=str(args.category or "all"),
        limit=max(1, int(args.limit)),
    )
    compact = [_compact_cli_row(r) for r in rows]
    if args.json:
        print(json.dumps(compact, indent=2, ensure_ascii=False))
        return 0

    if not compact:
        print("No style presets matched your query.")
        return 0

    print(f"Query: {args.query!r} | category={args.category} | top={len(compact)}")
    print("")
    for i, row in enumerate(compact, start=1):
        cmd = _format_as_command(row)
        print(f"{i:02d}. {row['id']}  [{row['category']}]  score={row['score']}")
        print(f"    {row['description']}")
        if cmd:
            print(f"    {cmd}")
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

