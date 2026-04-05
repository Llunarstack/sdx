#!/usr/bin/env python3
"""
Validate a JSON-serialized TrainConfig (or subset) against ``validate_train_config``.

Usage (repo root):
  python scripts/tools/dev/validate_config_json.py path/to/config.json
  python scripts/tools/dev/validate_config_json.py path/to/config.json --no-cuda-check

Exit code 0 if no ERROR lines; 1 if any ERROR; still prints WARNINGs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    p = argparse.ArgumentParser(description="Validate TrainConfig JSON")
    p.add_argument("config_json", type=Path, help="Path to JSON object with TrainConfig fields")
    p.add_argument(
        "--no-cuda-check",
        action="store_true",
        help="Treat missing CUDA as warning only (CI / laptop CPU)",
    )
    p.add_argument("--estimate-memory", action="store_true", help="Print rough VRAM estimate")
    args = p.parse_args()

    if not args.config_json.is_file():
        print(f"ERROR: file not found: {args.config_json}", file=sys.stderr)
        return 1

    raw = json.loads(args.config_json.read_text(encoding="utf-8"))
    from config.train_config import TrainConfig
    from utils.training.config_validator import estimate_memory_usage, validate_train_config

    cfg = TrainConfig(**raw)
    issues = validate_train_config(cfg, require_cuda=not args.no_cuda_check)
    for line in issues:
        print(line)

    if args.estimate_memory:
        est = estimate_memory_usage(cfg)
        print("\nMemory (rough):")
        for k, v in est.items():
            print(f"  {k}: {v}")

    errors = [x for x in issues if x.startswith("ERROR:")]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
