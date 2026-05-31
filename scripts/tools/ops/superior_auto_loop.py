#!/usr/bin/env python3
"""Full auto-improve loop with Superior Stack defaults (ViT mine, RAG, soup, composite pick)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--work-dir", default="superior_auto_loop")
    p.add_argument("--vit-ckpt", default="", help="ViT quality checkpoint for mining/scoring")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--model-soup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from utils.superior.auto_loop import AutoImproveConfig, run_auto_improve

    cfg = AutoImproveConfig(
        base_ckpt=args.base_ckpt,
        work_dir=args.work_dir,
        vit_ckpt=args.vit_ckpt,
        local_rag_jsonl=args.local_rag_jsonl,
        iterations=args.iterations,
        model_soup=bool(args.model_soup),
        pick_best="superior_composite",
        num=4,
    )
    return run_auto_improve(cfg, repo_root=_REPO, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
