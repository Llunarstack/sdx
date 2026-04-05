from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+", flags=re.ASCII)


def tokenize_normalized(text: str) -> List[str]:
    """
    Lightweight tokenizer for prompt adherence linting.

    - case-insensitive
    - alnum-only tokens
    - used for overlap detection and token-set heuristics (not for exact model tokenization)
    """
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _get_str(d: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None


def get_pos_neg_text(row: dict) -> Tuple[str, str]:
    """
    Extract (caption/positive, negative) from a manifest row.
    Keys match SDX conventions:
      - positive: caption | text
      - negative: negative_caption | negative_prompt | negative_text
    """
    pos = _get_str(row, ("caption", "text")) or ""
    neg = _get_str(row, ("negative_caption", "negative_prompt", "negative_text")) or ""
    return pos, neg


@dataclass(frozen=True)
class PromptLintOptions:
    min_caption_len_chars: int = 0
    max_caption_tokens: int = 0  # distinct token set size heuristic
    top_overlap_tokens: int = 10
    fail_on_overlap: bool = False


def lint_manifest_rows(rows: Iterable[dict], opts: PromptLintOptions) -> Dict[str, object]:
    total_lines = 0
    parse_errors = 0
    empty_caption_rows = 0
    json_missing_pos = 0
    rows_over_max_tokens = 0

    ok_rows = 0
    caption_token_count_sum = 0
    caption_token_count_min = 2**31 - 1
    caption_token_count_max = 0

    overlap_rows = 0
    max_overlap_distinct_tokens = 0
    overlap_token_counts: Dict[str, int] = {}

    for row in rows:
        total_lines += 1
        pos, neg = get_pos_neg_text(row)
        if not pos or (opts.min_caption_len_chars > 0 and len(pos) < opts.min_caption_len_chars):
            empty_caption_rows += 1
            continue
        if not _TOKEN_RE.search(pos):
            json_missing_pos += 1
            continue

        pos_tokens = tokenize_normalized(pos)
        neg_tokens = tokenize_normalized(neg)
        pos_set = set(pos_tokens)
        neg_set = set(neg_tokens)

        tok_len = len(pos_set)
        ok_rows += 1
        caption_token_count_sum += tok_len
        caption_token_count_min = min(caption_token_count_min, tok_len)
        caption_token_count_max = max(caption_token_count_max, tok_len)

        if opts.max_caption_tokens > 0 and tok_len > opts.max_caption_tokens:
            rows_over_max_tokens += 1

        if pos_set and neg_set:
            overlap = pos_set.intersection(neg_set)
            if overlap:
                overlap_rows += 1
                overlap_n = len(overlap)
                max_overlap_distinct_tokens = max(max_overlap_distinct_tokens, overlap_n)
                for tok in overlap:
                    overlap_token_counts[tok] = overlap_token_counts.get(tok, 0) + 1

    top = sorted(overlap_token_counts.items(), key=lambda kv: kv[1], reverse=True)
    if opts.top_overlap_tokens > 0:
        top = top[: opts.top_overlap_tokens]

    out: Dict[str, object] = {
        "promptlint_file": None,
        "lines_total": total_lines,
        "json_parse_errors": parse_errors,
        "empty_caption_rows": empty_caption_rows,
        "json_missing_pos_tokens": json_missing_pos,
        "rows_over_max_tokens": rows_over_max_tokens,
        "rows_ok": ok_rows,
        "caption_token_count_set_size_avg": (caption_token_count_sum / ok_rows) if ok_rows else 0.0,
        "caption_token_count_set_size_min": caption_token_count_min if ok_rows else 0,
        "caption_token_count_set_size_max": caption_token_count_max if ok_rows else 0,
        "pos_neg_overlap_rows": overlap_rows,
        "pos_neg_overlap_max_distinct_tokens": max_overlap_distinct_tokens,
        "top_overlap_tokens": top,
        "fail_on_overlap": opts.fail_on_overlap,
    }
    return out


def lint_jsonl_path(path: Path, opts: PromptLintOptions) -> Dict[str, object]:
    """
    Stream a JSONL file and compute prompt lint stats.
    Expects one JSON object per non-empty line.
    """
    total_lines = 0
    json_parse_errors = 0

    empty_caption_rows = 0
    json_missing_pos_tokens = 0
    rows_over_max_tokens = 0

    ok_rows = 0
    caption_token_count_sum = 0
    caption_token_count_min = 2**31 - 1
    caption_token_count_max = 0

    overlap_rows = 0
    max_overlap_distinct_tokens = 0
    overlap_token_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except Exception:
                json_parse_errors += 1
                continue

            if not isinstance(row, dict):
                json_parse_errors += 1
                continue

            pos, neg = get_pos_neg_text(row)
            if not pos or (opts.min_caption_len_chars > 0 and len(pos) < opts.min_caption_len_chars):
                empty_caption_rows += 1
                continue
            if not _TOKEN_RE.search(pos):
                json_missing_pos_tokens += 1
                continue

            pos_set = set(tokenize_normalized(pos))
            neg_set = set(tokenize_normalized(neg))

            tok_len = len(pos_set)
            ok_rows += 1
            caption_token_count_sum += tok_len
            caption_token_count_min = min(caption_token_count_min, tok_len)
            caption_token_count_max = max(caption_token_count_max, tok_len)

            if opts.max_caption_tokens > 0 and tok_len > opts.max_caption_tokens:
                rows_over_max_tokens += 1

            if pos_set and neg_set:
                overlap = pos_set.intersection(neg_set)
                if overlap:
                    overlap_rows += 1
                    overlap_n = len(overlap)
                    max_overlap_distinct_tokens = max(max_overlap_distinct_tokens, overlap_n)
                    for tok in overlap:
                        overlap_token_counts[tok] = overlap_token_counts.get(tok, 0) + 1

    top = sorted(overlap_token_counts.items(), key=lambda kv: kv[1], reverse=True)
    if opts.top_overlap_tokens > 0:
        top = top[: opts.top_overlap_tokens]

    return {
        "promptlint_file": str(path),
        "lines_total": total_lines,
        "json_parse_errors": json_parse_errors,
        "empty_caption_rows": empty_caption_rows,
        "json_missing_pos_tokens": json_missing_pos_tokens,
        "rows_over_max_tokens": rows_over_max_tokens,
        "rows_ok": ok_rows,
        "caption_token_count_set_size_avg": (caption_token_count_sum / ok_rows) if ok_rows else 0.0,
        "caption_token_count_set_size_min": caption_token_count_min if ok_rows else 0,
        "caption_token_count_set_size_max": caption_token_count_max if ok_rows else 0,
        "pos_neg_overlap_rows": overlap_rows,
        "pos_neg_overlap_max_distinct_tokens": max_overlap_distinct_tokens,
        "top_overlap_tokens": top,
        "fail_on_overlap": opts.fail_on_overlap,
    }


def should_fail(stats: Dict[str, object], opts: PromptLintOptions) -> bool:
    if not opts.fail_on_overlap:
        return False
    return int(stats.get("pos_neg_overlap_rows", 0)) > 0
