#!/usr/bin/env python3
"""
Book/comic/manga training launcher around root ``train.py``.

Prefer calling ``train.py`` directly with the same presets (one unified model path)::

    python train.py --manifest-jsonl data/book.jsonl --results-dir results/book \\
      --book-train-preset balanced --book-ar-profile comic_snake

This wrapper adds native manifest preflight and manifest_gate before spawning train.py.
Extra ``train.py`` args go after ``--``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.book_comic.training import (  # noqa: E402
    build_train_command,
    resolve_book_train_settings,
    run_native_manifest_preflight,
)

AR_PROFILES = (
    "auto",
    "none",
    "layout",
    "strong",
    "zorder",
    "vit_layout",
    "vit_strong",
    "comic_snake",
    "cinema_spiral",
)


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split *argv* at the first ``--`` into (wrapper args, raw ``train.py`` args)."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def add_manifest_gate_args(p: argparse.ArgumentParser) -> None:
    """The ``--manifest-gate-*`` tuning options, shared with ``prepare_and_train_book.py``."""
    p.add_argument("--manifest-gate-image-qc", action="store_true", help="Include image sharpness/contrast QC in gate.")
    p.add_argument(
        "--manifest-gate-image-root", type=str, default="", help="Optional base dir for image paths during QC."
    )
    p.add_argument("--manifest-gate-sample", type=int, default=0, help="Image QC sample size (0=all).")
    p.add_argument("--manifest-gate-min-sharpness", type=float, default=0.0)
    p.add_argument("--manifest-gate-min-contrast", type=float, default=0.0)
    p.add_argument("--manifest-gate-max-caption-tokens", type=int, default=0)
    p.add_argument("--manifest-gate-fail-on-overlap", action="store_true")
    p.add_argument("--manifest-gate-report-dups", action="store_true")
    p.add_argument("--manifest-gate-fail-on-dup-groups", type=int, default=0)


def manifest_gate_command(args: argparse.Namespace, manifest: Path) -> list[str]:
    """``python -m scripts.tools manifest_gate`` command for *manifest* built from the gate options in *args*."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.tools",
        "manifest_gate",
        str(manifest),
        "--min-caption-len-chars",
        str(args.manifest_min_caption_len),
        "--max-caption-tokens",
        str(args.manifest_gate_max_caption_tokens),
    ]
    if args.manifest_gate_fail_on_overlap:
        cmd.append("--fail-on-overlap")
    if args.manifest_gate_report_dups:
        cmd.append("--report-dups")
    if args.manifest_gate_fail_on_dup_groups > 0:
        cmd += ["--fail-on-dup-groups", str(args.manifest_gate_fail_on_dup_groups)]
    if args.manifest_gate_image_qc:
        cmd.append("--image-qc")
        image_root = args.manifest_gate_image_root.strip()
        if image_root:
            cmd += ["--image-root", image_root]
        if args.manifest_gate_sample > 0:
            cmd += ["--sample", str(args.manifest_gate_sample)]
        if args.manifest_gate_min_sharpness > 0:
            cmd += ["--min-sharpness", str(args.manifest_gate_min_sharpness)]
        if args.manifest_gate_min_contrast > 0:
            cmd += ["--min-contrast", str(args.manifest_gate_min_contrast)]
    return cmd


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
        choices=AR_PROFILES,
        help="One-flag AR setup: classic (layout/strong/zorder) or ViT-aligned (vit_layout/vit_strong/comic_snake/cinema_spiral).",
    )
    p.add_argument("--num-ar-blocks", type=int, default=-1, choices=[-1, 0, 2, 4], help="Explicit AR blocks override.")
    p.add_argument(
        "--ar-block-order",
        type=str,
        default="",
        choices=["", "raster", "zorder", "snake", "spiral"],
        help="Explicit AR block order override (use with --num-ar-blocks > 0).",
    )
    p.add_argument(
        "--ar-curriculum-mode",
        type=str,
        default="none",
        choices=["none", "step", "linear"],
        help="Runtime AR curriculum mode forwarded to train.py.",
    )
    p.add_argument("--ar-curriculum-warmup-steps", type=int, default=0)
    p.add_argument("--ar-curriculum-ramp-start", type=int, default=0)
    p.add_argument("--ar-curriculum-ramp-end", type=int, default=0)
    p.add_argument("--ar-curriculum-start-blocks", type=int, default=-1, choices=[-1, 0, 2, 4])
    p.add_argument("--ar-curriculum-target-blocks", type=int, default=-1, choices=[-1, 0, 2, 4])
    p.add_argument(
        "--ar-order-mix",
        type=str,
        default="",
        help="Comma list of AR orders for deterministic step-cycling (e.g. raster,zorder,snake).",
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

    p.add_argument(
        "--manifest-gate",
        action="store_true",
        help="Run manifest_gate (prompt lint + hygiene + optional image QC) before training.",
    )
    add_manifest_gate_args(p)
    p.add_argument("--resume", type=str, default="", help="Forward to train.py --resume (optimizer + step counter).")
    p.add_argument(
        "--init-from",
        type=str,
        dest="init_from",
        default="",
        help="Forward to train.py --init-from: load general SDX weights, start a fresh book fine-tune (step 0).",
    )
    return p


def _print_preflight(info: dict[str, Any]) -> None:
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


def main() -> int:
    own_argv, passthrough_train_args = split_passthrough(sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(own_argv)

    manifest_arg = args.manifest_jsonl.strip()
    if not args.data_path.strip() and not manifest_arg:
        parser.error("Provide at least one of --data-path or --manifest-jsonl.")
    if args.resume.strip() and args.init_from.strip():
        parser.error("Use either --resume or --init-from for book training, not both.")

    if args.native_preflight and manifest_arg:
        info = run_native_manifest_preflight(
            Path(manifest_arg),
            min_caption_len=args.manifest_min_caption_len,
            max_caption_len=args.manifest_max_caption_len,
        )
        _print_preflight(info)
        if args.strict_native and info["rust_validate_ok"] is False:
            print("[book_train] strict-native enabled: aborting due to failed Rust manifest validate.")
            return 2

    if args.manifest_gate and manifest_arg:
        gate_cmd = manifest_gate_command(args, Path(manifest_arg))
        print("[book_train] Manifest gate")
        print(" ", " ".join(gate_cmd))
        if subprocess.run(gate_cmd, cwd=ROOT).returncode != 0:
            print("[book_train] manifest_gate failed; aborting.")
            return 2

    cmd = build_train_command(
        root=ROOT,
        python_exe=sys.executable,
        args=args,
        settings=resolve_book_train_settings(args),
        passthrough_train_args=passthrough_train_args,
    )
    print("[book_train] Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
