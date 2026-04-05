"""
Pure-Python JSONL manifest **stats** and **prompt-lint** (replaces former ``native/js/*.mjs``).

CLI::

    python -m sdx_native.jsonl_manifest_pure stat path/to/manifest.jsonl
    python -m sdx_native.jsonl_manifest_pure promptlint path/to/manifest.jsonl \\
        [--min-caption-len-chars N] [--max-caption-tokens N] [--top-overlap-tokens N] [--fail-on-overlap]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _row_image_caption(obj: Mapping[str, Any]) -> Tuple[str, str]:
    image = str(obj.get("image_path") or obj.get("path") or obj.get("image") or "").strip()
    cap = str(obj.get("caption") or obj.get("text") or "").strip()
    return image, cap


def jsonl_stat_text(path: Path) -> str:
    """Same summary lines as the old ``sdx-jsonl-stat.mjs``."""
    total = empty_skip = parse_err = missing = ok = 0
    cap_lens: List[int] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            t = line.strip()
            if not t:
                empty_skip += 1
                continue
            try:
                obj = json.loads(t)
            except json.JSONDecodeError:
                parse_err += 1
                continue
            if not isinstance(obj, dict):
                parse_err += 1
                continue
            image, cap = _row_image_caption(obj)
            if image and cap:
                ok += 1
                cap_lens.append(len(cap))
            else:
                missing += 1
    cap_lens.sort()

    def pct(q: float) -> int:
        if not cap_lens:
            return 0
        idx = round((len(cap_lens) - 1) * q)
        idx = min(max(idx, 0), len(cap_lens) - 1)
        return cap_lens[idx]

    lines = [
        f"file: {path}",
        f"lines_total: {total}",
        f"empty_skipped: {empty_skip}",
        f"json_parse_errors: {parse_err}",
        f"rows_missing_image_or_caption: {missing}",
        f"rows_ok: {ok}",
    ]
    if cap_lens:
        lines.append(
            "caption_len_chars: "
            f"min={cap_lens[0]} p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={cap_lens[-1]}"
        )
    return "\n".join(lines) + "\n"


def _tokenize_normalized(text: Optional[str]) -> List[str]:
    out: List[str] = []
    cur = []
    for ch in str(text or ""):
        o = ord(ch)
        is_alnum = (48 <= o <= 57) or (65 <= o <= 90) or (97 <= o <= 122)
        if is_alnum:
            cur.append(ch.lower())
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _caption_neg_from_obj(obj: Dict[str, Any]) -> Tuple[str, str]:
    cap = None
    if isinstance(obj.get("caption"), str):
        cap = obj["caption"]
    elif isinstance(obj.get("text"), str):
        cap = obj["text"]
    else:
        cap = ""
    neg = None
    for k in ("negative_caption", "negative_prompt", "negative_text"):
        if isinstance(obj.get(k), str):
            neg = obj[k]
            break
    if neg is None:
        neg = ""
    return str(cap or ""), str(neg or "")


def promptlint_text(
    path: Path,
    *,
    min_caption_len_chars: int = 0,
    max_caption_tokens: int = 0,
    top_overlap_tokens: int = 10,
    fail_on_overlap: bool = False,
) -> Tuple[str, int]:
    """Same report lines and exit semantics as ``sdx-promptlint.mjs``."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    file_lines = raw.splitlines()

    total_lines = parse_errors = empty_caption_rows = rows_over_max_tokens = 0
    ok_rows = 0
    caption_token_set_sum = 0
    caption_token_set_min: Optional[int] = None
    caption_token_set_max = 0
    overlap_rows = 0
    max_overlap_distinct_tokens = 0
    overlap_token_counts: Dict[str, int] = {}

    for line in file_lines:
        total_lines += 1
        t = line.strip()
        if not t:
            continue
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(obj, dict):
            parse_errors += 1
            continue

        caption, neg = _caption_neg_from_obj(obj)
        if not caption or (min_caption_len_chars > 0 and len(caption) < min_caption_len_chars):
            empty_caption_rows += 1
            continue

        pos_tokens = _tokenize_normalized(caption)
        neg_tokens = _tokenize_normalized(neg)
        pos_set = set(pos_tokens)
        neg_set = set(neg_tokens)
        pos_tok_len = len(pos_set)
        ok_rows += 1
        caption_token_set_sum += pos_tok_len
        caption_token_set_min = (
            pos_tok_len if caption_token_set_min is None else min(caption_token_set_min, pos_tok_len)
        )
        caption_token_set_max = max(caption_token_set_max, pos_tok_len)

        if max_caption_tokens > 0 and pos_tok_len > max_caption_tokens:
            rows_over_max_tokens += 1

        if pos_set and neg_set:
            overlap = 0
            for tok in pos_set:
                if tok in neg_set:
                    overlap += 1
                    overlap_token_counts[tok] = overlap_token_counts.get(tok, 0) + 1
            if overlap > 0:
                overlap_rows += 1
                max_overlap_distinct_tokens = max(max_overlap_distinct_tokens, overlap)

    items = sorted(overlap_token_counts.items(), key=lambda x: -x[1])
    top = items[: top_overlap_tokens] if top_overlap_tokens > 0 else items

    out_lines = [
        f"promptlint: file {path}",
        f"lines_total: {total_lines}",
        f"json_parse_errors: {parse_errors}",
        f"empty_caption_rows: {empty_caption_rows}",
        f"rows_over_max_tokens: {rows_over_max_tokens}",
        f"rows_ok: {ok_rows}",
    ]
    if ok_rows > 0:
        avg = caption_token_set_sum / ok_rows
        cmin = caption_token_set_min if caption_token_set_min is not None else 0
        out_lines.append(f"caption_token_count(set_size): avg={avg:.2f} min={cmin} max={caption_token_set_max}")
    out_lines.extend(
        [
            f"pos_neg_overlap_rows: {overlap_rows}",
            f"pos_neg_overlap_max_distinct_tokens: {max_overlap_distinct_tokens}",
        ]
    )
    if top:
        out_lines.append("top_overlap_tokens: " + ", ".join(f"{tok}({cnt})" for tok, cnt in top))

    code = 1 if (fail_on_overlap and overlap_rows > 0) else 0
    return "\n".join(out_lines) + "\n", code


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m sdx_native.jsonl_manifest_pure stat|promptlint ...", file=sys.stderr)
        return 2
    cmd = argv.pop(0).lower()
    if cmd == "stat":
        if not argv:
            print("usage: python -m sdx_native.jsonl_manifest_pure stat <manifest.jsonl>", file=sys.stderr)
            return 2
        p = Path(argv[0])
        if not p.is_file():
            print(f"not a file: {p}", file=sys.stderr)
            return 2
        sys.stdout.write(jsonl_stat_text(p))
        return 0
    if cmd == "promptlint":
        ap = argparse.ArgumentParser(prog="jsonl_manifest_pure promptlint")
        ap.add_argument("file", type=Path)
        ap.add_argument("--min-caption-len-chars", type=int, default=0)
        ap.add_argument("--max-caption-tokens", type=int, default=0)
        ap.add_argument("--top-overlap-tokens", type=int, default=10)
        ap.add_argument("--fail-on-overlap", action="store_true")
        args, rest = ap.parse_known_args(argv)
        if rest:
            print(f"unknown arguments: {rest}", file=sys.stderr)
            return 2
        if not args.file.is_file():
            print(f"not a file: {args.file}", file=sys.stderr)
            return 2
        text, code = promptlint_text(
            args.file,
            min_caption_len_chars=args.min_caption_len_chars,
            max_caption_tokens=args.max_caption_tokens,
            top_overlap_tokens=args.top_overlap_tokens,
            fail_on_overlap=args.fail_on_overlap,
        )
        sys.stdout.write(text)
        return code
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
