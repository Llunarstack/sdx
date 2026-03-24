#!/usr/bin/env python3
"""
Designer → Verifier pipeline (LANDSCAPE §2): generate K candidates, ``--pick-best``, save winner.

Wraps ``sample.py`` (subprocess) so you get a single command for multi-sample + scoring.
Does not run a second refinement pass by default; pass ``--refine`` to keep sample.py defaults.

Example::

    python -m scripts.tools orchestrate_pipeline --ckpt results/000-DiT-XL-2-Text/best.pt \\
        --prompt \"a neon sign reading HELLO\" --num 4 --pick-best combo_exposure --out out.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    sample_py = root / "sample.py"
    p = argparse.ArgumentParser(description="Run sample.py with K candidates and pick-best.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="", dest="negative_prompt")
    p.add_argument("--out", type=str, default="orch_out.png")
    p.add_argument("--num", type=int, default=4, help="Number of candidates (>=2 for pick-best)")
    p.add_argument(
        "--pick-best",
        type=str,
        default="combo",
        choices=["none", "clip", "edge", "ocr", "combo", "combo_exposure"],
        dest="pick_best",
    )
    p.add_argument("--expected-text", type=str, default="", help="For ocr / combo when text-in-image")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--pick-save-all", action="store_true", help="Save all candidates as stem_cand{i}")
    p.add_argument("--no-refine", action="store_true", help="Pass --no-refine to sample.py")
    p.add_argument("--extra", nargs="*", default=[], help="Extra args passed through to sample.py")
    args = p.parse_args()

    cmd = [
        sys.executable,
        str(sample_py),
        "--ckpt",
        args.ckpt,
        "--prompt",
        args.prompt,
        "--out",
        args.out,
        "--num",
        str(max(1, args.num)),
        "--pick-best",
        args.pick_best,
        "--steps",
        str(args.steps),
        "--cfg-scale",
        str(args.cfg_scale),
    ]
    if args.negative_prompt:
        cmd += ["--negative-prompt", args.negative_prompt]
    if args.seed >= 0:
        cmd += ["--seed", str(args.seed)]
    if args.expected_text:
        cmd += ["--expected-text", args.expected_text]
    if args.pick_save_all:
        cmd.append("--pick-save-all")
    if args.no_refine:
        cmd.append("--no-refine")
    cmd.extend(args.extra)

    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
