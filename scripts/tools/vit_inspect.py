#!/usr/bin/env python3
"""
Inspect a ViT quality/adherence checkpoint: config, parameter count, optional module tree.

Usage:
    python -m scripts.tools vit_inspect path/to/vit_quality/best.pt
    python -m scripts.tools vit_inspect path/to/best.pt --tree --tree-depth 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect ViT quality checkpoint (params + config).")
    p.add_argument("ckpt", type=str, help="Path to ViT best.pt / last.pt")
    p.add_argument("--json", action="store_true", help="Print config as JSON")
    p.add_argument("--tree", action="store_true", help="Print module tree (see utils/nn_inspect.py)")
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--use-ema", action="store_true", help="Load EMA weights if present")
    args = p.parse_args()

    from utils.modeling.nn_inspect import format_module_tree
    from vit_quality.checkpoint_utils import load_vit_quality_checkpoint, vit_model_parameter_report

    path = Path(args.ckpt)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    model, cfg = load_vit_quality_checkpoint(path, use_ema=args.use_ema)
    rep = vit_model_parameter_report(model)

    if args.json:
        out = dict(cfg)
        out["total_parameters"] = rep["total_parameters"]
        out["trainable_parameters"] = rep["trainable_parameters"]
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Checkpoint: {path.resolve()}")
    print(f"Parameters: {rep['total_parameters']:,} total, {rep['trainable_parameters']:,} trainable")
    print("Config:")
    for k in sorted(cfg.keys()):
        print(f"  {k}: {cfg[k]}")

    if args.tree:
        print("\nModule tree:")
        for line in format_module_tree(model, max_depth=int(args.tree_depth)):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
