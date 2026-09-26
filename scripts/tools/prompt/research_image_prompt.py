#!/usr/bin/env python3
"""Research the best diffusion prompt for an image (VLM + local RAG, no SauceNAO).

python scripts/research_image_prompt.py path/to/image.png
python scripts/research_image_prompt.py path/to/image.png --rag-corpus D:\\Development\\sdx-data\\rag_corpus.jsonl
python scripts/research_image_prompt.py path/to/image.png --seed "1girl, fantasy" --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.caption.prompt_research import research_prompt_for_image  # noqa: E402


def _default_corpus() -> str:
    data = os.environ.get("SDX_DATA", r"D:\Development\sdx-data")
    return os.environ.get("SDX_RAG_CORPUS", str(Path(data) / "rag_corpus.jsonl"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Research diffusion prompt via uncensored VLM + local RAG (recommended over SauceNAO)."
    )
    p.add_argument("image", help="Image path")
    p.add_argument("--rag-corpus", default=_default_corpus(), help="rag_corpus.jsonl from build_rag_corpus.py")
    p.add_argument("--seed", default="", help="Optional intent seed merged with VLM description")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--creativity", type=float, default=0.45)
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--no-creative-rag", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    corpus = args.rag_corpus if Path(args.rag_corpus).is_file() else None
    result = research_prompt_for_image(
        args.image,
        rag_corpus=corpus,
        seed_prompt=args.seed,
        creativity_level=args.creativity,
        top_k=args.top_k,
        device=args.device,
        use_rag=not args.no_rag,
        use_creative_rag=not args.no_creative_rag,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "diffusion_prompt": result.diffusion_prompt,
                    "negative_prompt": result.negative_prompt,
                    "image_description": result.image_description,
                    "scene_summary": result.scene_summary,
                    "retrieved_facts": result.retrieved_facts,
                    "reasoning": result.reasoning,
                    "sources": result.sources,
                    "fallback_used": result.fallback_used,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("diffusion_prompt:", result.diffusion_prompt)
        print("negative_prompt:", result.negative_prompt)
        print("image_description:", result.image_description[:500])
        if result.retrieved_facts:
            print("rag_facts:", result.retrieved_facts[:3])
        print("reasoning:", result.reasoning[:400])
        print("sources:", result.sources)
        print("fallback:", result.fallback_used)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
