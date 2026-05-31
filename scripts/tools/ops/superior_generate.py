#!/usr/bin/env python3
"""
High-quality generation wrapper: local RAG + multi-candidate + composite pick-best.

Example::

    python -m scripts.tools superior_generate \\
        --ckpt results/run/best.pt \\
        --prompt "cyberpunk alley at night" \\
        --local-rag-jsonl datasets/style_notes.jsonl \\
        --num 4 --out out.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    p = argparse.ArgumentParser(description="Run sample.py with Superior Stack defaults.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="superior_out.png")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--local-rag-jsonl", default="", help="JSONL fact corpus for TF-IDF RAG")
    p.add_argument("--local-rag-top-k", type=int, default=8)
    p.add_argument("--self-correct", action="store_true")
    p.add_argument("--compile-inference", action="store_true")
    p.add_argument("--expand-prompt", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from utils.superior.inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv

    cfg = SuperiorInferenceConfig(
        num_candidates=max(1, args.num),
        local_rag_jsonl=args.local_rag_jsonl or None,
        rag_top_k=args.local_rag_top_k,
        self_correct_clip=bool(args.self_correct),
        compile_inference=bool(args.compile_inference),
        use_composite_rank=True,
    )
    argv = build_superior_sample_argv(ckpt=args.ckpt, prompt=args.prompt, out=args.out, config=cfg)
    if args.expand_prompt:
        argv.append("--expand-prompt")
    cmd = [sys.executable, str(_REPO / "sample.py"), *argv]
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=str(_REPO), check=True)


if __name__ == "__main__":
    main()
