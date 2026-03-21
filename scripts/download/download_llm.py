#!/usr/bin/env python3
"""
Download an LLM for prompt understanding/expansion (e.g. short user prompt → detailed caption).
Uses Hugging Face Hub. Use --best for top-quality (Qwen2.5-7B-Instruct); default is fast/small (SmolLM2-360M).
"""

import argparse
import os
import sys

# Repo root (scripts/download/ -> root)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BEST_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Best quality for instruction/prompt following at 7B
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Fast download + fast inference


def main():
    parser = argparse.ArgumentParser(
        description="Download an LLM for prompt understanding. Use --best for best quality (Qwen2.5-7B)."
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Download the best-quality model: Qwen2.5-7B-Instruct (top instruction following, ~15GB)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model ID (default: fast 360M; use --best for Qwen2.5-7B-Instruct)",
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
        help="Save to this dir (e.g. ./model/SmolLM2-360M-Instruct). Default: HF cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel download workers (default 8 for fast download)",
    )
    args = parser.parse_args()
    model = BEST_MODEL if args.best else args.model

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    kwargs = {
        "repo_id": model,
        "max_workers": args.max_workers,
    }
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    print(f"Downloading {model} (max_workers={args.max_workers})...")
    path = snapshot_download(**kwargs)
    print(f"Done. Model at: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
