#!/usr/bin/env python3
"""Git --msg-filter helper: drop Cursor attribution lines from commit messages."""
from __future__ import annotations

import sys

_SKIP = (
    "Made-with: Cursor",
    "Co-authored-by: Cursor",
    "cursoragent@cursor.com",
)


def main() -> None:
    lines = [
        ln
        for ln in sys.stdin.read().splitlines()
        if not any(s in ln for s in _SKIP)
    ]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
