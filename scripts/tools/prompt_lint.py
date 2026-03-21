#!/usr/bin/env python3
"""
Prompt adherence / dataset quality linting for SDX JSONL manifests.

Checks:
- missing / too-short captions
- token-length heuristic (distinct token set size via alnum tokenizer)
- positive/negative token overlap (pos_set ∩ neg_set)

This is a fast, dependency-light lint (not exact model tokenizer behavior).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from utils.prompt_lint import PromptLintOptions, lint_jsonl_path, should_fail

    p = argparse.ArgumentParser(description="Prompt lint for SDX JSONL manifests")
    p.add_argument("input", type=str, help="Path to JSONL manifest (one JSON object per line)")
    p.add_argument(
        "--min-caption-len-chars", type=int, default=0, help="Drop captions shorter than N characters (0=off)"
    )
    p.add_argument(
        "--max-caption-tokens", type=int, default=0, help="Max distinct-token-set size for positive captions (0=off)"
    )
    p.add_argument("--top-overlap-tokens", type=int, default=10, help="Print top overlapping tokens (0=off)")
    p.add_argument("--fail-on-overlap", action="store_true", help="Exit non-zero if any pos/neg overlap rows exist")
    p.add_argument("--json-report", type=str, default="", help="Optional path to write full stats JSON")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2

    opts = PromptLintOptions(
        min_caption_len_chars=int(args.min_caption_len_chars),
        max_caption_tokens=int(args.max_caption_tokens),
        top_overlap_tokens=int(args.top_overlap_tokens),
        fail_on_overlap=bool(args.fail_on_overlap),
    )
    stats = lint_jsonl_path(inp, opts)

    if args.json_report:
        outp = Path(args.json_report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"promptlint: file {stats['promptlint_file']}")
    print(f"lines_total: {stats['lines_total']}")
    print(f"json_parse_errors: {stats['json_parse_errors']}")
    print(f"empty_caption_rows: {stats['empty_caption_rows']}")
    print(f"json_missing_pos_tokens: {stats['json_missing_pos_tokens']}")
    print(f"rows_over_max_tokens: {stats['rows_over_max_tokens']}")
    print(f"rows_ok: {stats['rows_ok']}")
    print(
        "caption_token_count(set_size): "
        f"avg={stats['caption_token_count_set_size_avg']:.2f} "
        f"min={stats['caption_token_count_set_size_min']} max={stats['caption_token_count_set_size_max']}"
    )
    print(f"pos_neg_overlap_rows: {stats['pos_neg_overlap_rows']}")
    print(f"pos_neg_overlap_max_distinct_tokens: {stats['pos_neg_overlap_max_distinct_tokens']}")
    top = stats.get("top_overlap_tokens") or []
    if top:
        top_str = ", ".join(f"{tok}({cnt})" for tok, cnt in top)
        print(f"top_overlap_tokens: {top_str}")

    return 1 if should_fail(stats, opts) else 0


if __name__ == "__main__":
    sys.exit(main())
