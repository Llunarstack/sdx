#!/usr/bin/env python3
"""
Split a loose script into **one prompt per line** for ``generate_book.py`` ``--prompts-file``.

Supported separators (start of line):

- ``## Page 1`` / ``### Page 2`` (markdown headings)
- ``---PAGE---`` (dash line)

Blank lines inside a section are kept (folded to single spaces per output line unless ``--multiline``).

Usage::

    python scripts/tools/book_scene_split.py story.md --out pages.txt
    python pipelines/book_comic/scripts/generate_book.py ... --prompts-file pages.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List


def split_into_page_prompts(raw: str) -> List[str]:
    lines = raw.splitlines()
    pages: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        block = "\n".join(buf).strip()
        if block:
            pages.append(block)
        buf = []

    heading_re = re.compile(r"^#{1,3}\s*page\s*\d+\s*$", re.I)
    sep_re = re.compile(r"^---+PAGE---+|^-{3,}\s*page\s*break\s*-{3,}\s*$", re.I)

    for ln in lines:
        s = ln.strip()
        if heading_re.match(s) or sep_re.match(s):
            flush()
            continue
        buf.append(ln)

    flush()
    return pages


def normalize_one_line(text: str) -> str:
    """Collapse internal newlines to spaces for a single-line prompts file."""
    return " ".join(text.split())


def main() -> int:
    p = argparse.ArgumentParser(description="Split script into one line per page for generate_book.")
    p.add_argument("input", type=Path, help="Markdown or text file")
    p.add_argument("--out", type=Path, required=True, help="Output text file (one prompt per line)")
    args = p.parse_args()

    raw = args.input.read_text(encoding="utf-8", errors="ignore")
    pages = split_into_page_prompts(raw)
    if not pages:
        print("No pages found. Use '## Page 1' or ---PAGE--- between sections.", file=__import__("sys").stderr)
        return 1

    out_lines = [normalize_one_line(pg) for pg in pages]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} page line(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
