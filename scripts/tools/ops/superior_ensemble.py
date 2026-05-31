#!/usr/bin/env python3
"""Generate from multiple checkpoints; pick global best with CompositeRanker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="ensemble_best.png")
    p.add_argument("--num-per-ckpt", type=int, default=2)
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from utils.superior.ensemble import EnsembleConfig, generate_ensemble

    cfg = EnsembleConfig(
        checkpoints=args.checkpoints,
        num_per_ckpt=max(1, args.num_per_ckpt),
        vit_ckpt=args.vit_ckpt,
        local_rag_jsonl=args.local_rag_jsonl,
    )
    best, listing = generate_ensemble(args.prompt, cfg, out=args.out, repo_root=_REPO, dry_run=bool(args.dry_run))
    if best:
        print(f"Best: {best}")
        for ckpt, path, score in listing:
            print(f"  {ckpt}: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
