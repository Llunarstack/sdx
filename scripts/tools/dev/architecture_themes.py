#!/usr/bin/env python3
"""Dump ``utils.architecture.architecture_map`` themes as JSON or Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.architecture.architecture_map import summary_table_md, themes_as_dict  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--format",
        choices=("json", "md"),
        default="json",
        help="Output JSON rows or a Markdown summary table.",
    )
    args = ap.parse_args()
    if args.format == "json":
        print(json.dumps(themes_as_dict(), indent=2))
    else:
        print(summary_table_md())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
