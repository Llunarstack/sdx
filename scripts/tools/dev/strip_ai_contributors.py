#!/usr/bin/env python3
"""Strip AI Co-authored-by / Made-with trailers from every commit in the current repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strip_ai_git_trailers import filter_message  # noqa: E402


def message_callback(message: bytes) -> bytes:
    return filter_message(message.decode("utf-8", errors="replace")).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{__doc__} Rewrites every commit hash (git filter-repo --force).")
    ap.add_argument("--yes", action="store_true", help="Required: confirm rewriting the whole history")
    if not ap.parse_args(argv).yes:
        print("Refusing to rewrite history without --yes.", file=sys.stderr)
        return 2
    try:
        import git_filter_repo as fr
    except ImportError:
        print("ERROR: pip install git-filter-repo", file=sys.stderr)
        return 1

    args = fr.FilteringOptions.parse_args(["--force"])
    fr.RepoFilter(args, message_callback=message_callback).run()
    print("Done. Verify: git log --all -i --grep=co-authored-by")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
