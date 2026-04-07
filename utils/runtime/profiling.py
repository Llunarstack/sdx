"""Optional cProfile wrapper: strip profiling flags before argparse, then run the real entrypoint."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

__all__ = ["ProfileConfig", "consume_profile_args", "run_with_cprofile"]


@dataclass(frozen=True)
class ProfileConfig:
    out_path: str
    sort_key: str
    top_n: int


def consume_profile_args(argv: Sequence[str]) -> tuple[list[str], ProfileConfig | None]:
    """Remove ``--profile-out`` / ``--profile-sort`` / ``--profile-top`` so the main parser stays unchanged."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-out", default="", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "tottime", "ncalls", "name"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--profile-top", type=int, default=80, metavar="N", help=argparse.SUPPRESS)
    if len(argv) < 2:
        return list(argv), None
    ns, rest = parser.parse_known_args(argv[1:])
    out = str(ns.profile_out or "").strip()
    if not out:
        return list(argv), None
    top_n = max(1, int(ns.profile_top))
    cfg = ProfileConfig(out_path=out, sort_key=str(ns.profile_sort), top_n=top_n)
    return [argv[0]] + rest, cfg


def run_with_cprofile(entry: Callable[[], None], cfg: ProfileConfig) -> None:
    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    try:
        entry()
    finally:
        prof.disable()
        prof.dump_stats(str(out))
        summary_path = Path(str(out) + ".txt")
        with summary_path.open("w", encoding="utf-8") as fh:
            stats = pstats.Stats(prof, stream=fh)
            stats.sort_stats(cfg.sort_key)
            stats.print_stats(cfg.top_n)
        print(f"cProfile: wrote {out} and {summary_path}", file=sys.stderr)
