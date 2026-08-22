#!/usr/bin/env python3
"""
Download an LLM for prompt understanding/expansion (e.g. short user prompt → detailed caption).
Uses Hugging Face Hub. Use --best for top-quality (Qwen3-14B); default is fast/small (SmolLM2-360M).
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root (scripts/download/ -> root)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BEST_MODEL = "Qwen/Qwen3-14B"
BEST_FOLDER = "Qwen3-14B"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def main():
    parser = argparse.ArgumentParser(
        description="Download an LLM for prompt understanding. Use --best for best quality (Qwen3-14B)."
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Download the best-quality model: Qwen3-14B (top instruction following)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model ID (default: fast 360M; use --best for Qwen3-14B)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to save the model (default: HF_HOME or ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Save to this dir (e.g. ./pretrained/SmolLM2-360M-Instruct). Default: HF cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel download workers (default 8 for fast download)",
    )
    args = parser.parse_args()
    model = BEST_MODEL if args.best else args.model
    local_dir = args.local_dir
    if local_dir is None and args.best:
        local_dir = os.path.join(ROOT, "pretrained", BEST_FOLDER)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    kwargs = {
        "repo_id": model,
        "max_workers": args.max_workers,
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    print(f"Downloading {model} (max_workers={args.max_workers})...")
    path = snapshot_download(**kwargs)
    print(f"Done. Model at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
