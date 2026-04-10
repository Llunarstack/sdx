#!/usr/bin/env python3
"""
Append a **page-range design patch** to a book visual-memory JSON file.

Use when a user asks to change a character's proportions, camera, costume, etc.
from page N onward — updates the same schema consumed by ``generate_book.py --visual-memory``.

Examples (PowerShell; adjust quoting for your shell)::

    python -m scripts.tools visual_memory_patch memory.json --entity hero --from-page 12 --patch-json "{}"
    python -m scripts.tools visual_memory_patch memory.json --page-extra --from-page 0 --to-page 2 --prompt "rain scene"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipelines.book_comic.visual_memory import load_visual_memory

    ap = argparse.ArgumentParser(description="Patch book visual_memory JSON (page-scoped overrides).")
    ap.add_argument("memory_json", type=Path, help="Path to visual memory JSON (modified in place).")
    ap.add_argument("--from-page", type=int, required=True, help="0-based start page (inclusive).")
    ap.add_argument("--to-page", type=int, default=None, help="0-based end page (inclusive); default = from-page.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--entity", type=str, default="", help="Entity id to patch (see JSON entities map).")
    g.add_argument(
        "--page-extra",
        action="store_true",
        help="Append a run-level page_patches extra_prompt instead of an entity override.",
    )
    ap.add_argument(
        "--patch-json",
        type=str,
        default="",
        help="JSON object merged into the entity for that page range (with --entity). Omit if using --patch-file.",
    )
    ap.add_argument(
        "--patch-file",
        type=Path,
        default=None,
        help="Path to a JSON file (object) merged into the entity; avoids shell quoting issues.",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Free prompt fragment for --page-extra.",
    )
    args = ap.parse_args()

    path = args.memory_json
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    mem = load_visual_memory(path)

    if args.page_extra:
        prompt = (args.prompt or "").strip()
        if not prompt:
            print("error: --page-extra requires non-empty --prompt", file=sys.stderr)
            return 2
        mem.add_page_patch(from_page=args.from_page, to_page=args.to_page, extra_prompt=prompt)
    else:
        eid = (args.entity or "").strip()
        if not eid:
            print("error: --entity required unless --page-extra", file=sys.stderr)
            return 2
        if args.patch_file is not None:
            if not args.patch_file.is_file():
                print(f"error: --patch-file not found: {args.patch_file}", file=sys.stderr)
                return 2
            try:
                patch = json.loads(args.patch_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"error: invalid JSON in --patch-file: {e}", file=sys.stderr)
                return 2
        else:
            raw = (args.patch_json or "").strip() or "{}"
            try:
                patch = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"error: invalid --patch-json: {e}", file=sys.stderr)
                return 2
        if not isinstance(patch, dict):
            print("error: patch must be a JSON object", file=sys.stderr)
            return 2
        mem.apply_entity_page_patch(eid, from_page=args.from_page, to_page=args.to_page, patch=patch)

    mem.save(path)
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
