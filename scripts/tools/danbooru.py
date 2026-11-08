#!/usr/bin/env python3
"""Danbooru tag vocabulary: fetch -> split general tags into buckets -> merge one categorized TSV.

    python -m scripts.tools danbooru fetch --out-dir data/danbooru/tags/raw --max-pages 5
    python -m scripts.tools danbooru split --general data/danbooru/tags/raw/general.txt
    python -m scripts.tools danbooru merge --out data/danbooru/tags/all_tags_categorized.txt
    python -m scripts.tools danbooru all --max-pages 10 --sleep 0.2

``fetch`` uses the public JSON API (https://danbooru.donmai.us/wiki_pages/help:api); categories
0 general, 1 artist, 3 copyright, 4 character, 5 meta. Most clothing / object / style tokens are
**general**, which ``split`` buckets with underscore-aware substring rules
(``data/danbooru/general_subsplit_rules.json``; first matching bucket in ``bucket_order`` wins).
``merge`` writes ``category<TAB>name<TAB>post_count<TAB>deprecated``: non-general raw rows plus all
bucket rows (buckets partition ``general.txt``; without buckets, ``general.txt`` is labeled ``general``).
``all`` runs the three steps; unknown flags are forwarded to ``fetch``.

Set ``DANBOORU_USERNAME`` + ``DANBOORU_API_KEY`` for HTTP basic auth (higher rate limits).
Old command names (``fetch_danbooru_tags``, ``split_danbooru_general_tags``,
``merge_danbooru_categorized_tags``, ``download_all_danbooru_categorized_tags``) still work.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BASE = "https://danbooru.donmai.us"
DEFAULT_UA = "sdx-danbooru-fetch/1.0 (tag export; respect rate limits)"
CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}
RAW_DIR = Path("data/danbooru/tags/raw")
BUCKETS_DIR = Path("data/danbooru/tags/buckets")
RULES = Path("data/danbooru/general_subsplit_rules.json")
MERGED = Path("data/danbooru/tags/all_tags_categorized.txt")


def _request(url: str, *, ua: str, auth_header: str | None, timeout: float = 120.0, retries: int = 4) -> Any:
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
        if auth_header:
            req.add_header("Authorization", auth_header)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt + 1 >= retries:
                raise
        except OSError:
            if attempt + 1 >= retries:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _basic_auth_header() -> str | None:
    user = (os.environ.get("DANBOORU_USERNAME") or "").strip()
    key = (os.environ.get("DANBOORU_API_KEY") or "").strip()
    if not user or not key:
        return None
    return "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode("ascii")


def fetch_pages(
    *, limit: int, max_pages: int | None, sleep_s: float, skip_deprecated: bool, ua: str, auth_header: str | None
) -> defaultdict[int, list[str]]:
    """category_id -> lines ``name<TAB>post_count<TAB>deprecated``."""
    buckets: defaultdict[int, list[str]] = defaultdict(list)
    page = 1
    while max_pages is None or page <= max_pages:
        q = urllib.parse.urlencode({"limit": str(limit), "page": str(page), "search[order]": "id"})
        data = _request(f"{BASE}/tags.json?{q}", ua=ua, auth_header=auth_header)
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if not isinstance(row, dict) or (skip_deprecated and row.get("is_deprecated")):
                continue
            cat, name = row.get("category"), row.get("name")
            if cat not in CATEGORY_NAMES or not name:
                continue
            dep = "1" if row.get("is_deprecated") else "0"
            buckets[int(cat)].append(f"{name}\t{row.get('post_count', 0)}\t{dep}\n")
        page += 1
        if len(data) < limit:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return buckets


def _fetch_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.tools danbooru fetch", description="Fetch all Danbooru tags into category files."
    )
    ap.add_argument("--out-dir", type=Path, default=RAW_DIR, help="Output directory")
    ap.add_argument("--limit", type=int, default=1000, help="Tags per page (max 1000)")
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: until empty)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Delay between pages (seconds)")
    ap.add_argument("--skip-deprecated", action="store_true", help="Skip is_deprecated tags")
    ap.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent string")
    args = ap.parse_args(argv)

    lim = min(1000, max(1, args.limit))
    auth = _basic_auth_header()
    if auth:
        print("Using DANBOORU_USERNAME / DANBOORU_API_KEY for auth", file=sys.stderr)
    buckets = fetch_pages(
        limit=lim,
        max_pages=args.max_pages,
        sleep_s=max(0.0, args.sleep),
        skip_deprecated=bool(args.skip_deprecated),
        ua=args.user_agent,
        auth_header=auth,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for cat_id, name in CATEGORY_NAMES.items():
        lines = buckets.get(cat_id, [])
        path = args.out_dir / f"{name}.txt"
        path.write_text("".join(lines), encoding="utf-8")
        print(f"{name} ({cat_id}): {len(lines)} tags -> {path}", file=sys.stderr)
        total += len(lines)
    print(f"total lines written: {total}", file=sys.stderr)
    return 0


def _wrap(s: str) -> str:
    """Normalize for token-ish matching: ``_foo_bar_``."""
    return "_" + s.lower().replace("-", "_").replace(" ", "_").strip("_") + "_"


def matches_pattern(tag: str, pattern: str) -> bool:
    return _wrap(pattern) in _wrap(tag)


def load_rules(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["bucket_order"]), {k: list(v) for k, v in data["patterns"].items()}


def split_general(general_path: Path, rules_path: Path, out_dir: Path) -> dict[str, int]:
    order, patterns = load_rules(rules_path)
    wrapped = [(b, [_wrap(p) for p in patterns.get(b, [])]) for b in order]
    buckets: dict[str, list[str]] = {b: [] for b in order}
    buckets["general_other"] = []
    with general_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            tag = line.split("\t", 1)[0].strip()
            if not tag:
                continue
            wt = _wrap(tag)
            dest = next((b for b, pats in wrapped if any(p in wt for p in pats)), "general_other")
            buckets[dest].append(line + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, lines in buckets.items():
        p = out_dir / f"{name}.txt"
        p.write_text("".join(lines), encoding="utf-8")
        counts[name] = len(lines)
        print(f"{name}: {len(lines)} -> {p}", file=sys.stderr)
    return counts


def _split_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.tools danbooru split", description="Bucket Danbooru general tags for training."
    )
    ap.add_argument("--general", type=Path, required=True, help="Path to general.txt from `danbooru fetch`")
    ap.add_argument("--rules", type=Path, default=RULES, help="JSON with bucket_order and patterns")
    ap.add_argument("--out-dir", type=Path, default=BUCKETS_DIR, help="Output directory")
    args = ap.parse_args(argv)
    if not args.general.is_file():
        print(f"Missing {args.general}", file=sys.stderr)
        return 1
    if not args.rules.is_file():
        print(f"Missing {args.rules}", file=sys.stderr)
        return 2
    split_general(args.general, args.rules, args.out_dir)
    return 0


def _iter_lines(path: Path) -> Iterable[str]:
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            yield from (line.rstrip("\n") for line in f if line.strip())


def merge_categorized(raw_dir: Path, buckets_dir: Path, out: Path) -> int:
    sources = [(cat, raw_dir / f"{cat}.txt") for cat in ("artist", "copyright", "character", "meta")]
    bucket_txts = (
        sorted(p for p in buckets_dir.glob("*.txt") if not p.name.startswith(".")) if buckets_dir.is_dir() else []
    )
    sources += [(p.stem, p) for p in bucket_txts] or [("general", raw_dir / "general.txt")]
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for cat, path in sources:
            for line in _iter_lines(path):
                f.write(f"{cat}\t{line}\n")
                n += 1
    return n


def _merge_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.tools danbooru merge", description="Merge categorized Danbooru tag files into one TSV."
    )
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--buckets-dir", type=Path, default=BUCKETS_DIR)
    ap.add_argument("--out", type=Path, default=MERGED)
    args = ap.parse_args(argv)
    n = merge_categorized(args.raw_dir, args.buckets_dir, args.out)
    print(f"Wrote {n} lines to {args.out}")
    return 0


def _all_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.tools danbooru all",
        description="Fetch Danbooru tags and build categorized export (unknown flags go to `fetch`).",
    )
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw tag output directory")
    ap.add_argument("--buckets-dir", type=Path, default=BUCKETS_DIR, help="Bucket output directory")
    ap.add_argument("--rules", type=Path, default=RULES, help="General-tag split rules")
    ap.add_argument("--merged-out", type=Path, default=MERGED, help="Single combined TSV output")
    args, fetch_extra = ap.parse_known_args(argv)
    rc = _fetch_main(["--out-dir", str(args.raw_dir), *fetch_extra])
    if rc:
        return rc
    rc = _split_main(
        ["--general", str(args.raw_dir / "general.txt"), "--rules", str(args.rules), "--out-dir", str(args.buckets_dir)]
    )
    if rc:
        return rc
    return _merge_main(
        ["--raw-dir", str(args.raw_dir), "--buckets-dir", str(args.buckets_dir), "--out", str(args.merged_out)]
    )


COMMANDS = {"fetch": _fetch_main, "split": _split_main, "merge": _merge_main, "all": _all_main}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
