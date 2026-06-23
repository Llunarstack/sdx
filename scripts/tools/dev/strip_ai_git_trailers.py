#!/usr/bin/env python3
"""Git --msg-filter helper: drop AI attribution lines from commit messages."""

from __future__ import annotations

import re
import sys

_SKIP_SUBSTRINGS = (
    "made-with: cursor",
    "co-authored-by: cursor",
    "cursoragent@cursor.com",
    "cursoragent@users.noreply.github.com",
    "noreply@anthropic.com",
    "anthropic.com",
)

_SKIP_PATTERNS = (
    re.compile(r"^co-authored-by:\s*claude\b", re.IGNORECASE),
    re.compile(r"^co-authored-by:\s*cursor\b", re.IGNORECASE),
    re.compile(r"^made-with:\s*cursor\b", re.IGNORECASE),
)


def should_drop(line: str) -> bool:
    lowered = line.lower()
    if any(sub in lowered for sub in _SKIP_SUBSTRINGS):
        return True
    return any(pattern.match(line.strip()) for pattern in _SKIP_PATTERNS)


def filter_message(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not should_drop(ln)]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.write(filter_message(sys.stdin.read()))


if __name__ == "__main__":
    main()
