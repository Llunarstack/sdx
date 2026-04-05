#!/usr/bin/env python3
"""
Book/comic/manga training launcher around root ``train.py``.

Adds:
- Preset bundles tuned for sequential art.
- Optional native preflight over JSONL manifests (Rust/Zig when available).
- Pass-through support for any additional train.py flags after ``--``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.book_comic.book_training_helpers import (  # noqa: E402
    build_train_command,
    resolve_book_train_settings,
    run_native_manifest_preflight,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Book/comic/manga model trainer wrapper around train.py.")
    p.add_argument("--data-path", type=str, default="", help="Image folder root (optional if --manifest-jsonl is set).")
    p.add_argument("--manifest-jsonl", type=str, default="", help="JSONL dataset manifest for training.")
    p.add_argument("--results-dir", type=str, default="results/book_train")

    p.add_argument("--book-train-preset", type=str, default="balanced", choices=["fast", "balanced", "production"])
    p.add_argument("--model", type=str, default="", help="Override preset model name.")
    p.add_argument("--image-size", type=int, default=0, help="Override preset image size.")
    p.add_argument("--global-batch-size", type=int, default=0, help="Override preset global batch size.")
    p.add_argument("--lr", type=float, default=0.0, help="Override preset learning rate.")
    p.add_argument("--passes", type=int, default=-1, help="Override preset passes (>=0).")
    p.add_argument("--max-steps", type=int, default=-1, help="Override preset max-steps (>=0).")
    p.add_argument(
        "--ar-profile",
        type=str,
        default="auto",
        choices=["auto", "none", "layout", "strong", "zorder"],
        help="One-flag AR setup: layout=2x2 raster, strong=4x4 raster, zorder=2x2 z-order.",
    )
    p.add_argument("--num-ar-blocks", type=int, default=-1, choices=[-1, 0, 2, 4], help="Explicit AR blocks override.")
    p.add_argument(
        "--ar-block-order",
        type=str,
        default="",
        choices=["", "raster", "zorder"],
        help="Explicit AR block order override (use with --num-ar-blocks > 0).",
    )

    p.add_argument("--num-workers", type=int, default=-1, help="If >=0, forward to train.py --num-workers.")
    p.add_argument("--dry-run", action="store_true", help="Forward train.py --dry-run.")
    p.add_argument("--no-compile", action="store_true", help="Forward train.py --no-compile.")
    p.add_argument("--no-xformers", action="store_true", help="Forward train.py --no-xformers.")

    p.add_argument(
        "--native-preflight",
        action="store_true",
        help="Run native manifest diagnostics before training (Rust validate/stats + Zig/Python fingerprint).",
    )
    p.add_argument(
        "--strict-native",
        action="store_true",
        help="Fail when native preflight is requested and Rust validator returns non-zero.",
    )
    p.add_argument("--manifest-min-caption-len", type=int, default=0)
    p.add_argument("--manifest-max-caption-len", type=int, default=0)
    return p


def main() -> int:
    raw = sys.argv[1:]
    if "--" in raw:
        i = raw.index("--")
        own_argv = raw[:i]
        passthrough_train_args = raw[i + 1 :]
    else:
        own_argv = raw
        passthrough_train_args = []

    parser = _build_parser()
    args = parser.parse_args(own_argv)

    if not str(args.data_path).strip() and not str(args.manifest_jsonl).strip():
        parser.error("Provide at least one of --data-path or --manifest-jsonl.")

    if args.native_preflight and str(args.manifest_jsonl).strip():
        manifest = Path(args.manifest_jsonl)
        info = run_native_manifest_preflight(
            manifest,
            min_caption_len=int(args.manifest_min_caption_len),
            max_caption_len=int(args.manifest_max_caption_len),
        )
        print("[book_train] Native preflight")
        print(f"  manifest: {info['manifest']}")
        print(f"  exists: {info['exists']}")
        if info["fingerprint"]:
            print(f"  fingerprint: {info['fingerprint']}")
        print(f"  rust_validate_ok: {info['rust_validate_ok']}")
        if info["rust_validate_stderr"]:
            print(f"  rust_validate_stderr: {info['rust_validate_stderr']}")
        print(f"  rust_stats_ok: {info['rust_stats_ok']}")
        if info["rust_stats_stdout"]:
            print("  rust_stats_stdout:")
            print(info["rust_stats_stdout"])
        if args.strict_native and info["rust_validate_ok"] is False:
            print("[book_train] strict-native enabled: aborting due to failed Rust manifest validate.")
            return 2

    settings = resolve_book_train_settings(args)
    cmd = build_train_command(
        root=ROOT,
        python_exe=sys.executable,
        args=args,
        settings=settings,
        passthrough_train_args=passthrough_train_args,
    )
    print("[book_train] Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
