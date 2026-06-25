#!/usr/bin/env python3
"""Append benchmark_suite leaderboard snapshots to rolling history."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.generation.sample_features import append_benchmark_history


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leaderboard", type=Path, default=Path("benchmark_out/leaderboard.json"))
    p.add_argument("--history", type=Path, default=Path("benchmark_out/history.json"))
    args = p.parse_args()
    if not args.leaderboard.is_file():
        p.error(f"leaderboard not found: {args.leaderboard}")
    entry = append_benchmark_history(args.leaderboard, args.history)
    print(f"Appended snapshot at {entry['timestamp']} -> {args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
