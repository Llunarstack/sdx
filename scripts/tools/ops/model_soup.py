#!/usr/bin/env python3
"""Average multiple DiT checkpoints into one soup weights file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--template", required=True, help="Checkpoint for config/metadata template")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from utils.superior.model_soup import save_soup_checkpoint, soup_checkpoints

    avg = soup_checkpoints(args.checkpoints, weights=args.weights)
    save_soup_checkpoint(avg, template_path=args.template, out_path=args.out)
    print(f"Saved model soup ({len(args.checkpoints)} ckpts) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
