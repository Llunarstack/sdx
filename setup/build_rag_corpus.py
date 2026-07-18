#!/usr/bin/env python3
"""Build a local RAG corpus JSONL from scraped training manifests.

At inference, point ``--local-rag-jsonl`` at this file. SDX runs TF-IDF retrieval
over caption/tag text from your dataset and merges top facts into the prompt
before encoding (see ``utils/prompt/rag_prompt.py``).

    python setup/build_rag_corpus.py --data-root /workspace/data
    python sample.py --prompt "1girl in a field" --local-rag-jsonl /workspace/data/rag_corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.sites import ADAPTERS  # noqa: E402


def _build_corpus_impl(manifest_paths: list[Path], out_path: Path, *, max_entries: int) -> int:
    seen: set[str] = set()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for mp in manifest_paths:
            if not mp.is_file():
                continue
            for line in mp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if written >= max_entries:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cap = (row.get("caption") or "").strip()
                summary = (row.get("scene_summary") or "").strip()
                if not cap and not summary:
                    continue
                key = (cap + "|" + summary).lower()
                if key in seen:
                    continue
                seen.add(key)
                artists = row.get("artist_tags") or []
                chars = row.get("character_tags") or []
                series = row.get("copyright_tags") or []
                parts = []
                if chars:
                    parts.append(f"characters: {', '.join(chars)}")
                if series:
                    parts.append(f"series: {', '.join(series)}")
                if artists:
                    parts.append(f"artist: {', '.join(artists)}")
                if summary:
                    parts.append(summary)
                if cap:
                    parts.append(f"tags: {cap}")
                text = ". ".join(parts) if parts else cap
                out_f.write(
                    json.dumps({"text": text, "caption": cap, "source": row.get("source", "")}, ensure_ascii=False)
                    + "\n"
                )
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build RAG corpus JSONL from scrape manifests.")
    p.add_argument("--data-root", default=None, help="Scan combined + per-site manifests under this dir.")
    p.add_argument("--manifest", action="append", default=[], help="Explicit manifest path (repeatable).")
    p.add_argument("--out", default=None, help="Output path (default: <data-root>/rag_corpus.jsonl).")
    p.add_argument("--max-entries", type=int, default=500_000)
    p.add_argument("--sites", nargs="*", default=sorted(ADAPTERS))
    args = p.parse_args(argv)

    manifests = [Path(m) for m in args.manifest]
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None:
        combined = data_root / "combined" / "manifest.jsonl"
        if combined.is_file():
            manifests.append(combined)
        for site in args.sites:
            mp = data_root / site / "manifest.jsonl"
            if mp.is_file():
                manifests.append(mp)
    manifests = list(dict.fromkeys(manifests))
    if not manifests:
        print("No manifests found.", file=sys.stderr)
        return 2

    out = (
        Path(args.out)
        if args.out
        else (data_root / "rag_corpus.jsonl" if data_root else REPO_ROOT / "data" / "rag_corpus.jsonl")
    )
    n = _build_corpus_impl(manifests, out, max_entries=int(args.max_entries))
    print(f"Wrote {n:,} RAG entries -> {out}")
    print(f"Use at inference: python sample.py --local-rag-jsonl {out} --prompt '...'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
