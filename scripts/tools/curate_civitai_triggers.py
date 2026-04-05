#!/usr/bin/env python3
"""
Rebuild Civitai-derived tag files from the model bank CSV:

- ``data/civitai/top_triggers_by_frequency.txt`` — one token per line, most common first
- ``utils/prompt/civitai_vocab.py`` — ``CIVITAI_HOT_TAGS`` (top 150) for ``content_controls``

Run from repo root after ``fetch_civitai_nsfw_concepts.py``::

    python scripts/tools/curate_civitai_triggers.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _frequency_lists(csv_path: Path, *, hot_n: int, freq_lines: int) -> tuple[list[str], list[str]]:
    ctr: Counter[str] = Counter()
    first_canon: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for t in (row.get("triggers") or "").split("|"):
                x = t.strip()
                if len(x) < 2 or len(x) > 100:
                    continue
                k = x.lower()
                ctr[k] += 1
                if k not in first_canon:
                    first_canon[k] = x
    pairs = sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))
    ordered = [first_canon[k] for k, _ in pairs]
    hot = ordered[:hot_n]
    freq_file = ordered[:freq_lines]
    return hot, freq_file


def _write_vocab_py(out: Path, hot: list[str]) -> None:
    lines = [
        '"""',
        "Top trigger tokens from ``data/civitai/nsfw_illustrious_noobai_models.csv`` (frequency-ranked).",
        "",
        "Regenerate after refreshing the CSV::",
        "",
        "    python scripts/tools/curate_civitai_triggers.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# Regenerate: python scripts/tools/curate_civitai_triggers.py",
        "CIVITAI_HOT_TAGS = [",
    ]
    for t in hot:
        lines.append(f"    {t!r},")
    lines.append("]")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate Civitai CSV triggers into txt + civitai_vocab.py")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("data/civitai/nsfw_illustrious_noobai_models.csv"),
        help="Input model bank CSV",
    )
    ap.add_argument(
        "--freq-out",
        type=Path,
        default=Path("data/civitai/top_triggers_by_frequency.txt"),
        help="Output: one trigger per line (frequency order)",
    )
    ap.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("utils/prompt/civitai_vocab.py"),
        help="Output: CIVITAI_HOT_TAGS Python module",
    )
    ap.add_argument("--hot-n", type=int, default=150, help="How many tags to embed in civitai_vocab.py")
    ap.add_argument(
        "--freq-lines",
        type=int,
        default=1200,
        help="How many lines for top_triggers_by_frequency.txt",
    )
    ap.add_argument(
        "--names-out",
        type=Path,
        default=None,
        help="Optional: write model names one per line (same order as CSV rows)",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Missing CSV: {args.csv}")

    hot, freq = _frequency_lists(args.csv, hot_n=args.hot_n, freq_lines=args.freq_lines)
    args.freq_out.parent.mkdir(parents=True, exist_ok=True)
    args.freq_out.write_text("\n".join(freq), encoding="utf-8")
    _write_vocab_py(args.vocab_out, hot)
    print(f"Wrote {len(freq)} lines -> {args.freq_out}")
    print(f"Wrote CIVITAI_HOT_TAGS ({len(hot)} items) -> {args.vocab_out}")

    if args.names_out is not None:
        names: list[str] = []
        with args.csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n = (row.get("name") or "").strip()
                if n:
                    names.append(n)
        args.names_out.parent.mkdir(parents=True, exist_ok=True)
        args.names_out.write_text("\n".join(names), encoding="utf-8")
        print(f"Wrote {len(names)} model names -> {args.names_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
