#!/usr/bin/env python3
"""Write Markdown report from benchmark_suite output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bench_dir", help="Directory with results.json + leaderboard.json")
    p.add_argument("--out", default="", help="Output .md (default: bench_dir/REPORT.md)")
    args = p.parse_args()

    from utils.superior.eval_report import build_markdown_report, write_report

    bench = Path(args.bench_dir)
    out = Path(args.out) if args.out else bench / "REPORT.md"
    write_report(bench, out)
    print(build_markdown_report(bench))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
