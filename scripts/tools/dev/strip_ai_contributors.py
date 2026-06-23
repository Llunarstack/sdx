#!/usr/bin/env python3
"""Strip AI Co-authored-by trailers from all commits in the current repo."""

from __future__ import annotations

import re
import sys

_SKIP_SUB = (
    "made-with: cursor",
    "co-authored-by: cursor",
    "cursoragent@cursor.com",
    "cursoragent@users.noreply.github.com",
    "noreply@anthropic.com",
    "anthropic.com",
)
_PATTERNS = (
    re.compile(r"^co-authored-by:\s*claude\b", re.IGNORECASE),
    re.compile(r"^co-authored-by:\s*cursor\b", re.IGNORECASE),
    re.compile(r"^made-with:\s*cursor\b", re.IGNORECASE),
)


def _filter_message(text: str) -> bytes:
    lines: list[str] = []
    for ln in text.splitlines():
        low = ln.lower()
        if any(s in low for s in _SKIP_SUB):
            continue
        if any(p.match(ln.strip()) for p in _PATTERNS):
            continue
        lines.append(ln)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def message_callback(message: bytes) -> bytes:
    return _filter_message(message.decode("utf-8", errors="replace"))


def main() -> int:
    try:
        import git_filter_repo as fr
    except ImportError:
        print("ERROR: pip install git-filter-repo", file=sys.stderr)
        return 1

    args = fr.FilteringOptions.parse_args(["--force"])
    repo_filter = fr.RepoFilter(args, message_callback=message_callback)
    repo_filter.run()
    print("Done. Verify: git log --grep=Co-Authored -i")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
