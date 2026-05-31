#!/usr/bin/env python3
"""
Download Hugging Face model scaffolds (config + tokenizer only, no checkpoints).

Creates ``pretrained/<folder>`` trees so SDX wiring and ``pretrained_status`` can
see local repos. Runtime still uses the hub when weights are missing locally
(see ``resolve_model_path_require_weights`` in ``utils/modeling/model_paths.py``).

Examples:
    python scripts/download/download_hf_scaffold.py --all
    python scripts/download/download_hf_scaffold.py --role vlm --role reward
    python scripts/download/download_hf_scaffold.py --name Florence-2-base --name HPSv2-hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.modeling.hf_scaffold import (  # noqa: E402
    HF_SCAFFOLD_REGISTRY,
    download_scaffold_batch,
    resolve_entries,
    scaffold_registry,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Download HF config-only scaffolds into pretrained/.")
    p.add_argument("--model-dir", type=str, default="", help="Target root (default: pretrained/)")
    p.add_argument("--all", action="store_true", help="Download full scaffold registry")
    p.add_argument(
        "--role",
        action="append",
        default=[],
        help="Filter by role (text_encoder, vlm, reward, depth, ocr, detector, ...). Repeatable.",
    )
    p.add_argument("--name", action="append", default=[], help="Download specific catalog names. Repeatable.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Re-download even if folder exists")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first download error")
    p.add_argument("--list", action="store_true", help="List registry and exit")
    p.add_argument("--out-json", type=str, default="", help="Write download report JSON")
    p.add_argument(
        "--penta",
        action="store_true",
        help="Download penta text encoder scaffolds (T5-XXL, CLIP-L, bigG, H, LongCLIP-L).",
    )
    p.add_argument(
        "--triple",
        action="store_true",
        help="Download triple text encoder scaffolds (T5-XXL, CLIP-L, bigG).",
    )
    args = p.parse_args()

    if args.list:
        for e in scaffold_registry():
            print(f"{e.name}\t{e.role}\t{e.repo_id}")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        return 1

    if args.all:
        entries = list(HF_SCAFFOLD_REGISTRY)
    elif args.penta:
        from utils.modeling.text_encoder_stack import PENTA_CATALOG

        entries = resolve_entries(names=list(PENTA_CATALOG), roles=None)
    elif args.triple:
        from utils.modeling.text_encoder_stack import TRIPLE_CATALOG

        entries = resolve_entries(names=list(TRIPLE_CATALOG), roles=None)
    else:
        entries = resolve_entries(names=args.name or None, roles=args.role or None)

    if not entries:
        print("Choose --all, --role, or --name (use --list to see registry).", file=sys.stderr)
        return 1

    model_root = Path(args.model_dir) if str(args.model_dir).strip() else ROOT / "pretrained"
    report = download_scaffold_batch(
        entries,
        model_root=model_root,
        max_workers=int(args.max_workers),
        skip_existing=not bool(args.force),
        continue_on_error=not bool(args.fail_fast),
    )
    ok = list(report.get("ok", []))
    failed = list(report.get("failed", []))
    print(f"Done. scaffold_ok={len(ok)} failed={len(failed)} -> {model_root}")
    for msg in failed:
        print(f"  FAILED: {msg}", file=sys.stderr)

    if str(args.out_json).strip():
        out = Path(str(args.out_json).strip())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report -> {out}")

    return 1 if failed and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())
