#!/usr/bin/env python3
"""Enrich manifest captions with multi-source image profiling.

Default (``SDX_PROMPT_RESEARCH=1``): VLM + local RAG + Creative RAG (Qwen) produce
detailed diffusion prompts for training, with booru identity tags merged in.

Fast path: ``--booru-only`` or ``SDX_PROMPT_RESEARCH=0`` — restructures API tags only.

    python setup/enrich_manifest_captions.py \\
        --manifest /workspace/data/combined/manifest.jsonl \\
        --data-root /workspace/data \\
        --out /workspace/data/enriched/manifest.jsonl

Requires no API keys — SauceNAO/TinEye use the same web upload as a browser.
Booru credentials in ``secret.txt`` are always used for tag lookups after a match.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.caption.api_keys import reverse_search_enabled  # noqa: E402
from utils.caption.image_profiler import profile_from_manifest_row  # noqa: E402
from utils.caption.prompt_research import PromptResearchResult, research_prompt_for_image  # noqa: E402


def _row_needs_enrich(row: dict, *, use_prompt_research: bool) -> bool:
    if use_prompt_research:
        sources = [str(s) for s in (row.get("tag_sources") or [])]
        return not any(
            x in sources or any(x in s for s in sources)
            for x in ("creative_rag", "vlm_uncensored", "rag_fallback")
        )
    return not row.get("scene_summary")


def _merge_research_row(row: dict, researched: PromptResearchResult) -> dict:
    """Fuse booru identity tags with VLM+RAG+LLM diffusion prompt."""
    merged = dict(row)
    patch = researched.to_manifest_patch()
    cap = str(patch.get("caption") or "").strip()
    identity: list[str] = []
    for key in ("character_tags", "copyright_tags", "artist_tags"):
        for t in row.get(key) or []:
            t = str(t).strip().replace("_", " ")
            if t and t.lower() not in cap.lower():
                identity.append(t)
    if identity:
        patch["caption"] = ", ".join(identity) + (", " + cap if cap else "")
    if row.get("caption"):
        patch["booru_caption"] = row["caption"]
    sources = [str(s) for s in (row.get("tag_sources") or [])]
    sources.extend(str(s) for s in (patch.get("tag_sources") or []))
    patch["tag_sources"] = list(dict.fromkeys(sources))
    merged.update(patch)
    return merged


def enrich_manifest(
    manifest: Path,
    data_root: Path,
    out: Path,
    *,
    workers: int = 4,
    max_rows: int = 0,
    use_vlm: bool = True,
    use_reverse: bool = True,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    use_prompt_research: bool = False,
    rag_corpus: Path | None = None,
    skip_complete: bool = True,
) -> int:
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_rows > 0:
        rows = rows[:max_rows]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0

    reverse_on = use_reverse and reverse_search_enabled(use_saucenao=use_saucenao, use_tineye=use_tineye)
    corpus = Path(rag_corpus) if rag_corpus and Path(rag_corpus).is_file() else None
    if use_prompt_research:
        workers = 1  # VLM + LLM are GPU-serial; parallel workers OOM the pod

    def _one(row: dict) -> dict | None:
        if skip_complete and not _row_needs_enrich(row, use_prompt_research=use_prompt_research):
            return row
        try:
            if use_prompt_research:
                rel = row.get("image_path") or ""
                img = data_root / rel if rel else Path("_missing")
                if not img.is_file():
                    img = Path(rel) if rel else Path("_missing")
                if img.is_file():
                    seed = str(row.get("caption") or "")
                    researched = research_prompt_for_image(
                        img,
                        rag_corpus=corpus,
                        seed_prompt=seed,
                        use_rag=corpus is not None,
                    )
                    return _merge_research_row(row, researched)
            prof = profile_from_manifest_row(
                row,
                data_root,
                use_vlm=use_vlm,
                use_reverse_search=reverse_on,
                use_saucenao=use_saucenao,
                use_tineye=use_tineye,
            )
            merged = dict(row)
            merged.update(prof.to_manifest_patch())
            return merged
        except Exception:
            return row

    with open(tmp, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_one, r): i for i, r in enumerate(rows)}
            results: list[dict | None] = [None] * len(rows)
            for fut in futures:
                results[futures[fut]] = fut.result()
            for result in results:
                if result is None:
                    continue
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"enrichment produced 0 rows — not overwriting {out}")
    tmp.replace(out)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-source caption enrichment (booru + SauceNAO + VLM).")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--no-reverse-search", action="store_true")
    p.add_argument("--no-saucenao", action="store_true")
    p.add_argument("--no-tineye", action="store_true")
    p.add_argument(
        "--prompt-research",
        action="store_true",
        default=None,
        help="VLM + local RAG + Creative RAG (Qwen) for detailed training captions (default: SDX_PROMPT_RESEARCH=1).",
    )
    p.add_argument(
        "--booru-only",
        action="store_true",
        help="Fast path: booru tag restructuring only (no VLM/RAG/LLM).",
    )
    p.add_argument("--rag-corpus", default="", help="rag_corpus.jsonl for --prompt-research")
    p.add_argument("--force", action="store_true", help="Re-profile even when tags look complete.")
    args = p.parse_args(argv)

    import os

    if args.booru_only:
        use_prompt_research = False
    elif args.prompt_research is not None:
        use_prompt_research = bool(args.prompt_research)
    else:
        use_prompt_research = os.environ.get("SDX_PROMPT_RESEARCH", "1") == "1"

    rag = Path(args.rag_corpus) if args.rag_corpus else None
    if use_prompt_research and rag is None:
        data = Path(args.data_root)
        default_rag = Path(os.environ.get("SDX_RAG_CORPUS", data / "rag_corpus.jsonl"))
        rag = default_rag if default_rag.is_file() else None

    n = enrich_manifest(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.out),
        workers=args.workers,
        max_rows=args.max_rows,
        use_vlm=not args.no_vlm and not use_prompt_research,
        use_reverse=not args.no_reverse_search and not use_prompt_research,
        use_saucenao=not args.no_saucenao,
        use_tineye=not args.no_tineye,
        use_prompt_research=use_prompt_research,
        rag_corpus=rag,
        skip_complete=not args.force,
    )
    mode = "VLM+RAG+LLM prompt research" if use_prompt_research else "booru metadata"
    print(f"Wrote {n:,} rows -> {args.out} ({mode})")
    print("Point training at enriched manifest; scene_summary feeds RAG via build_rag_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
