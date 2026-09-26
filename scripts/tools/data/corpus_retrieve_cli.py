#!/usr/bin/env python3
"""Build / query a training-corpus retrieval index (board-scale JSONL).

Examples:
  python -m scripts.tools build_corpus_index --jsonl data/danbooru.jsonl --out data/corpus_index.jsonl
  python -m scripts.tools corpus_retrieve --index data/corpus_index.jsonl --prompt "1girl, catgirl, eipril"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Corpus index build / retrieve")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build index JSONL from train manifest")
    b.add_argument("--jsonl", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--root", default="")
    b.add_argument("--max-rows", type=int, default=0)
    b.add_argument("--require-files", action="store_true")

    q = sub.add_parser("query", help="Retrieve refs for a prompt")
    q.add_argument("--index", required=True)
    q.add_argument("--prompt", required=True)
    q.add_argument("--top-k", type=int, default=6)
    q.add_argument("--source", default="", help="Filter: danbooru|e621|gelbooru|...")
    q.add_argument("--moodboard-out", default="")
    q.add_argument("--root", default="")

    args = p.parse_args(argv)
    from utils.generation.corpus_retrieve import (
        build_corpus_index_from_jsonl,
        hits_to_moodboard_payload,
        retrieve_corpus_refs,
        save_corpus_index_meta,
    )

    if args.cmd == "build":
        idx = build_corpus_index_from_jsonl(
            args.jsonl,
            root=args.root or None,
            max_rows=int(args.max_rows),
            require_existing_file=bool(args.require_files),
        )
        out = save_corpus_index_meta(idx, args.out)
        print(f"indexed {len(idx)} rows -> {out}")
        return 0

    idx = build_corpus_index_from_jsonl(args.index, root=args.root or None)
    hits = retrieve_corpus_refs(idx, args.prompt, top_k=int(args.top_k), source_filter=args.source)
    print(json.dumps([h.to_dict() for h in hits], indent=2))
    if args.moodboard_out:
        payload = hits_to_moodboard_payload(hits)
        Path(args.moodboard_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"moodboard -> {args.moodboard_out} ({len(payload.get('images') or [])} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
