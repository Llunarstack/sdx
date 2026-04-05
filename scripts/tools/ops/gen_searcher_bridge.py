#!/usr/bin/env python3
"""
Convert Gen-Searcher-style JSON/JSONL outputs into SDX fact JSONL and optional merged prompt preview.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.modeling.model_paths import default_gen_searcher_8b_path, verify_gen_searcher_8b_local
from utils.prompt.rag_prompt import (
    load_facts_from_gen_searcher_json,
    merge_facts_into_prompt,
)


def _write_facts_jsonl(path: Path, facts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in facts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-json", type=str, required=True, help="Gen-Searcher output path (.json or .jsonl).")
    ap.add_argument("--out-facts-jsonl", type=str, default="gen_searcher_facts.jsonl")
    ap.add_argument("--max-facts", type=int, default=24)
    ap.add_argument("--max-item-chars", type=int, default=280)
    ap.add_argument("--prompt", type=str, default="", help="Optional prompt to produce merged prompt preview.")
    ap.add_argument("--out-merged-prompt", type=str, default="", help="Optional path to write merged prompt text.")
    ap.add_argument("--max-merge-chars", type=int, default=2400)
    ap.add_argument(
        "--gen-searcher-model-path",
        type=str,
        default=default_gen_searcher_8b_path(),
        help="Local Gen-Searcher-8B folder or HF repo id (used for verification/status output).",
    )
    ap.add_argument(
        "--verify-local-model",
        action="store_true",
        help="Verify required local Gen-Searcher shard files before conversion.",
    )
    args = ap.parse_args()

    if bool(args.verify_local_model):
        report = verify_gen_searcher_8b_local(str(args.gen_searcher_model_path))
        print(f"[gen_searcher_bridge] model path: {args.gen_searcher_model_path}")
        print(
            f"[gen_searcher_bridge] local_ok={report['is_local_dir']} all_required_present={report['all_required_present']}"
        )
        if report.get("missing"):
            print(f"[gen_searcher_bridge] missing: {report['missing']}")

    facts = load_facts_from_gen_searcher_json(
        args.in_json,
        max_entries=max(1, int(args.max_facts)),
        max_item_chars=max(64, int(args.max_item_chars)),
    )
    out_path = Path(args.out_facts_jsonl)
    _write_facts_jsonl(out_path, facts)
    print(f"[gen_searcher_bridge] wrote {len(facts)} facts -> {out_path}")

    if str(args.prompt).strip() and str(args.out_merged_prompt).strip():
        merged = merge_facts_into_prompt(
            str(args.prompt).strip(),
            facts,
            max_chars=max(256, int(args.max_merge_chars)),
        )
        mp = Path(args.out_merged_prompt)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(merged, encoding="utf-8")
        print(f"[gen_searcher_bridge] wrote merged prompt -> {mp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
