#!/usr/bin/env python3
"""
Audit a **prompts file** (one line per page) for a book run: length / token budget,
optional cast-name presence, optional visual-memory alignment.

Usage::

    python -m scripts.tools book_prompt_audit pages.txt
    python -m scripts.tools book_prompt_audit pages.txt --visual-memory memory.json --cast Ren Kai
    python -m scripts.tools book_prompt_audit pages.txt --warn-chars 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from pipelines.book_comic.book_prompt_intel import (
        approximate_token_estimate,
        composed_prompt_length_report,
        find_cast_mentions,
    )
    from pipelines.book_comic.visual_memory import load_visual_memory

    ap = argparse.ArgumentParser(description="Audit book page prompts (length, cast mentions).")
    ap.add_argument("prompts_file", type=Path, help="Text file: one page prompt per line")
    ap.add_argument(
        "--warn-chars",
        type=int,
        default=0,
        help="Warn when a line exceeds this many characters (0 = off).",
    )
    ap.add_argument(
        "--warn-tokens",
        type=int,
        default=0,
        help="Warn when approximate whole-line tokens exceed this (0 = off).",
    )
    ap.add_argument(
        "--cast",
        nargs="*",
        default=[],
        help="Display names that should appear in each page line (substring match).",
    )
    ap.add_argument(
        "--visual-memory",
        type=Path,
        default=None,
        help="Optional visual memory JSON: adds memory fragment to length report per page.",
    )
    ap.add_argument(
        "--consistency-chars",
        type=int,
        default=0,
        help="Simulate extra consistency block length added on every page (for budgeting).",
    )
    args = ap.parse_args()

    pf = args.prompts_file
    if not pf.is_file():
        print(f"error: not found: {pf}", file=sys.stderr)
        return 2

    lines = [ln.strip() for ln in pf.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        print("error: no non-empty lines", file=sys.stderr)
        return 2

    vm = None
    if args.visual_memory is not None:
        if not args.visual_memory.is_file():
            print(f"error: visual memory not found: {args.visual_memory}", file=sys.stderr)
            return 2
        vm = load_visual_memory(args.visual_memory)

    cast = [str(x).strip() for x in (args.cast or []) if str(x).strip()]
    fake_consistency = "x" * max(0, int(args.consistency_chars))

    rc = 0
    for i, page in enumerate(lines):
        vm_frag = vm.prompt_fragment_for_page(i) if vm is not None else ""
        rep = composed_prompt_length_report(
            user_prompt=page,
            consistency_block=fake_consistency,
            visual_memory_fragment=vm_frag,
        )
        tot = rep["total_chars"]
        toks = approximate_token_estimate(page + " " + vm_frag + " " + fake_consistency)
        miss_note = ""
        if cast:
            r = find_cast_mentions(page, cast)
            if r.missing:
                miss_note = f" missing_cast={r.missing}"
                rc = max(rc, 1)
        warn = ""
        if args.warn_chars and tot > int(args.warn_chars):
            warn += f" LONG>{args.warn_chars}"
            rc = max(rc, 1)
        if args.warn_tokens and toks > int(args.warn_tokens):
            warn += f" TOKS>{args.warn_tokens}"
            rc = max(rc, 1)
        print(f"page{i:04d} chars={tot} approx_line_tokens={toks}{miss_note}{warn}")

    print(f"summary: {len(lines)} page line(s)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
