#!/usr/bin/env python3
"""Run sample.py N times with seed offsets; keep best output by heuristic score."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from utils.generation.sample_features import run_auto_refine_candidates


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=str, default="refined.png")
    p.add_argument("--stem", type=str, default="refine_candidate")
    p.add_argument("sample_args", nargs=argparse.REMAINDER, help="Args passed to sample.py (after --)")
    args = p.parse_args()
    sample_args = list(args.sample_args)
    if sample_args and sample_args[0] == "--":
        sample_args = sample_args[1:]
    cmd = [sys.executable, str(Path(__file__).resolve().parents[2] / "sample.py")] + sample_args
    res = run_auto_refine_candidates(
        sample_cmd=cmd,
        num_candidates=args.candidates,
        seed_base=args.seed_base,
        out_stem=args.stem,
    )
    best = res.outputs[res.best_index]
    shutil.copy(best, args.out)
    print(f"Best candidate {res.best_index} score={res.scores[res.best_index]:.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
