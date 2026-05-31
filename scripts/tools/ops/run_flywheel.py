#!/usr/bin/env python3
"""Run the full Superior quality flywheel (curate + align + promote)."""

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
    p.add_argument("--work-dir", default="flywheel_run")
    p.add_argument("--manifest-in", default="", help="Training JSONL to curate")
    p.add_argument("--manifest-out", default="", help="Curated JSONL output")
    p.add_argument("--local-rag-jsonl", default="")
    p.add_argument("--vit-ckpt", default="")
    p.add_argument("--promote-path", default="")
    p.add_argument("--skip-curate", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from config.defaults.superior_stack import FlywheelPlan, SuperiorStackDefaults
    from utils.superior.flywheel import run_flywheel

    defaults = SuperiorStackDefaults(auto_loop_iterations=max(1, args.iterations))
    plan = FlywheelPlan(
        base_ckpt=args.base_ckpt,
        work_dir=args.work_dir,
        manifest_in=args.manifest_in or None,
        manifest_out=args.manifest_out or None,
        local_rag_jsonl=args.local_rag_jsonl or None,
        vit_ckpt=args.vit_ckpt or None,
        promote_path=args.promote_path or str(Path(args.work_dir) / "best.pt"),
        skip_curate=bool(args.skip_curate) or not args.manifest_in,
        skip_align=bool(args.skip_align),
        defaults=defaults,
    )
    summary = run_flywheel(plan, repo_root=_REPO, dry_run=bool(args.dry_run))
    print(summary)
    return 0 if summary.get("status") != "align_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
