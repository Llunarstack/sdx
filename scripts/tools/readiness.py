#!/usr/bin/env python3
"""Training-readiness gate (see training/readiness.py). Both commands take train.py's own arguments.

    python -m scripts.tools readiness dataset --data-path data/train [--limit 5000] [--max-bad-frac 0.01]
    python -m scripts.tools readiness check --data-path data/train --model DiT-XL/2-Text ... [--device cpu]

``dataset`` audits the data train.py reads (JSON with --json-out). ``check`` runs every check and writes a
READY / NOT READY report to ``<results-dir>/readiness/``; train.py requires a READY one for long runs.
Exit code 0 = PASS / READY.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROG = "python -m scripts.tools readiness"


def _train_cfg(rest: list[str]):
    from training.cli import apply_book_train_preset_to_args, build_train_arg_parser, build_train_config_from_args

    args = build_train_arg_parser().parse_args(rest)
    apply_book_train_preset_to_args(args)
    return build_train_config_from_args(args)


def _dataset_main(argv: list[str]) -> int:
    from data.audit import audit_dataset, format_audit

    p = argparse.ArgumentParser(
        prog=f"{_PROG} dataset", description="Audit the training dataset; unknown args go to train.py's parser."
    )
    p.add_argument("--limit", type=int, default=0, help="Audit a seeded random sample of N rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-bad-frac", type=float, default=0.01, help="FAIL above this fraction of dropped/unreadable rows"
    )
    p.add_argument("--min-caption-tokens", type=int, default=3)
    p.add_argument("--max-crop-loss", type=float, default=0.4, help="Flag rows whose crop drops more of the frame")
    p.add_argument("--no-loader-check", action="store_true", help="Skip running the dataset loader per row (faster)")
    p.add_argument("--json-out", type=str, default="", help="Write the full report here")
    args, rest = p.parse_known_args(argv)
    rep = audit_dataset(
        _train_cfg(rest),
        limit=args.limit,
        seed=args.seed,
        max_bad_frac=args.max_bad_frac,
        min_caption_tokens=args.min_caption_tokens,
        max_crop_loss=args.max_crop_loss,
        check_loader=not args.no_loader_check,
        progress=lambda k, n: print(f"  audited {k}/{n}", file=sys.stderr),
    )
    print(format_audit(rep))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report: {args.json_out}")
    return 0 if rep["ok"] else 1


def _check_main(argv: list[str]) -> int:
    from training.readiness import format_report, run_readiness

    p = argparse.ArgumentParser(
        prog=f"{_PROG} check", description="Run the readiness gate; unknown args go to train.py's parser."
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="cuda/auto also measures throughput"
    )
    p.add_argument("--audit-limit", type=int, default=5000, help="Rows sampled by the dataset audit (0 = all)")
    p.add_argument("--max-bad-frac", type=float, default=0.01)
    p.add_argument("--overfit-steps", type=int, default=400)
    p.add_argument("--overfit-samples", type=int, default=8)
    p.add_argument(
        "--overfit-drop",
        type=float,
        default=0.15,
        help="PASS when the fixed-noise eval loss drops this much. The tiny copy reaches ~0.9 (eps), ~0.7 (x0) but "
        "plateaus near 0.25 (v) / 0.35 (flow) within a few hundred steps; a broken objective stays ~0. "
        "Use --real-model with a higher bar for a capacity-level check.",
    )
    p.add_argument("--overfit-lr", type=float, default=0.0, help="0 = 1e-3 tiny model / --lr real model")
    p.add_argument("--real-model", action="store_true", help="Overfit the configured architecture, not a tiny copy")
    args, rest = p.parse_known_args(argv)
    rep = run_readiness(
        _train_cfg(rest),
        device=args.device,
        audit_limit=args.audit_limit,
        max_bad_frac=args.max_bad_frac,
        overfit_steps=args.overfit_steps,
        overfit_samples=args.overfit_samples,
        overfit_drop=args.overfit_drop,
        overfit_lr=args.overfit_lr,
        real_model=args.real_model,
        argv=rest,
    )
    print(format_report(rep))
    return 0 if rep["ready"] else 1


COMMANDS = {"dataset": _dataset_main, "check": _check_main}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
