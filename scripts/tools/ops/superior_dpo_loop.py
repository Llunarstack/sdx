#!/usr/bin/env python3
"""Mine benchmark pairs → train Diffusion-DPO in one command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-json", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pairs-jsonl", default="prefs_mined.jsonl")
    p.add_argument("--out", default="dpo_policy.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from utils.superior.dpo_pipeline import DPOStageConfig, MinePairsConfig, run_alignment_loop

    summary = run_alignment_loop(
        benchmark_json=args.benchmark_json,
        base_ckpt=args.ckpt,
        pairs_jsonl=args.pairs_jsonl,
        dpo_out=args.out,
        mine=MinePairsConfig(
            benchmark_json=args.benchmark_json,
            out_jsonl=args.pairs_jsonl,
            min_margin=args.min_margin,
        ),
        dpo=DPOStageConfig(ckpt=args.ckpt, preference_jsonl=args.pairs_jsonl, out=args.out, steps=args.steps),
        repo_root=_REPO,
        dry_run=args.dry_run,
    )
    print(summary)
    return int(summary.get("dpo_exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
