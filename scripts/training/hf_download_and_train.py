#!/usr/bin/env python3
"""
Download images + captions from a Hugging Face dataset, write SDX JSONL, then train a **basic** DiT.

This is a thin wrapper around:
  - ``scripts/training/hf_export_to_sdx_manifest.py`` (download + manifest)
  - ``train.py`` (same repo root)

The dataset **must** include an image column and a caption/tags column (see docs/DANBOORU_HF.md).
Use ``--list-columns`` on the export script first if unsure.

Usage:

    pip install datasets

    python scripts/training/hf_download_and_train.py \\
        --dataset YOUR_ORG/your-dataset \\
        --max-samples 500 \\
        --image-field image \\
        --caption-field tag_string

Optional: pass extra ``train.py`` arguments after ``--``:

    python scripts/training/hf_download_and_train.py --dataset X --max-samples 200 -- \\
        --max-steps 50 --dry-run

**Demo without Hugging Face** (synthetic local images only):

    python scripts/training/hf_download_and_train.py --demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# DataLoader workers > 0 often deadlocks on Windows; default to 0 there.
_DEFAULT_TRAIN_WORKERS = 0 if sys.platform == "win32" else 2

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    raw = sys.argv[1:]
    if "--" in raw:
        i = raw.index("--")
        export_argv = raw[:i]
        train_argv = raw[i + 1 :]
    else:
        export_argv = raw
        train_argv = []

    p = argparse.ArgumentParser(description="HF export + basic DiT training.")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Skip HF: use scripts/tools/make_smoke_dataset.py + train (no dataset download).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/hf_basic_run"))
    p.add_argument("--results-dir", type=Path, default=Path("results/hf_basic_run"))
    p.add_argument("--dataset", type=str, default="", help="HF dataset id (required unless --demo).")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--image-field", type=str, default="image")
    p.add_argument("--caption-field", type=str, default="tag_string")
    p.add_argument("--caption-tag-join", type=str, default=", ")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the full split into memory instead of streaming (can use a lot of RAM).",
    )
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--model", type=str, default="DiT-B/2-Text")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--global-batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=100, help="Passed to train.py (ignored if --dry-run).")
    p.add_argument("--dry-run", action="store_true", help="Only 1 training step.")
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"DataLoader workers for train.py (default: {_DEFAULT_TRAIN_WORKERS} on this OS).",
    )
    p.add_argument(
        "--no-xformers",
        action="store_true",
        help="Pass --no-xformers to train.py (helps some Windows/CUDA setups).",
    )
    args = p.parse_args(export_argv)

    py = sys.executable
    train_py = ROOT / "train.py"
    export_py = ROOT / "scripts" / "tr" / "hf_export_to_sdx_manifest.py"
    smoke_py = ROOT / "scripts" / "tools" / "make_smoke_dataset.py"

    if args.demo:
        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [py, str(smoke_py), "--out", str(out), "--count", "8"],
            cwd=ROOT,
        )
        if r.returncode != 0:
            return r.returncode
        # make_smoke_dataset writes under out/train/
        train_cmd = [
            py,
            str(train_py),
            "--data-path",
            str(out),
            "--results-dir",
            str(args.results_dir),
            "--model",
            args.model,
            "--image-size",
            str(args.image_size),
            "--global-batch-size",
            str(args.global_batch_size),
            "--no-compile",
            "--num-workers",
            "0",
        ]
        if args.dry_run:
            train_cmd.append("--dry-run")
        else:
            train_cmd.extend(["--max-steps", str(args.max_steps)])
        nw = _DEFAULT_TRAIN_WORKERS if args.num_workers is None else args.num_workers
        train_cmd[train_cmd.index("--num-workers") + 1] = str(nw)
        if args.no_xformers:
            train_cmd.append("--no-xformers")
        train_cmd.extend(train_argv)
        print("[hf_download_and_train] Running:", " ".join(train_cmd))
        return subprocess.run(train_cmd, cwd=ROOT).returncode

    if not args.dataset.strip():
        print("Error: pass --dataset YOUR_ORG/name or use --demo", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    streaming = not args.no_streaming

    exp_cmd = [
        py,
        str(export_py),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--image-field",
        args.image_field,
        "--caption-field",
        args.caption_field,
        "--caption-tag-join",
        args.caption_tag_join,
        "--out-dir",
        str(out_dir),
        "--manifest-name",
        args.manifest_name,
        "--max-samples",
        str(args.max_samples),
    ]
    if args.config:
        exp_cmd.extend(["--config", args.config])
    if args.revision:
        exp_cmd.extend(["--revision", args.revision])
    if streaming:
        exp_cmd.append("--streaming")
    if args.shuffle_seed is not None:
        exp_cmd.extend(["--shuffle-seed", str(args.shuffle_seed)])

    print("[hf_download_and_train] Export:", " ".join(exp_cmd))
    r = subprocess.run(exp_cmd, cwd=ROOT)
    if r.returncode != 0:
        return r.returncode

    manifest = out_dir / args.manifest_name
    train_cmd = [
        py,
        str(train_py),
        "--manifest-jsonl",
        str(manifest),
        "--data-path",
        str(out_dir),
        "--results-dir",
        str(args.results_dir),
        "--model",
        args.model,
        "--image-size",
        str(args.image_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--no-compile",
        "--num-workers",
        str(_DEFAULT_TRAIN_WORKERS if args.num_workers is None else args.num_workers),
    ]
    if args.dry_run:
        train_cmd.append("--dry-run")
    else:
        train_cmd.extend(["--max-steps", str(args.max_steps)])
    if args.no_xformers:
        train_cmd.append("--no-xformers")
    train_cmd.extend(train_argv)

    print("[hf_download_and_train] Train:", " ".join(train_cmd))
    return subprocess.run(train_cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
