#!/usr/bin/env python3
"""
Download Danbooru tag vocabulary via the public JSON API and split by official category.

API reference: https://danbooru.donmai.us/wiki_pages/help:api

Categories (integer ``category`` on each tag):
  0 general, 1 artist, 3 copyright, 4 character, 5 meta

Most clothing / object / style *tokens* live under **general** (0). Run
``split_danbooru_general_tags.py`` afterward to bucket those for training.

Respect Danbooru: use a descriptive User-Agent, keep ``--sleep`` reasonable, optional API key.

Examples::

    python scripts/tools/fetch_danbooru_tags.py --out-dir data/danbooru/tags/raw --max-pages 5
    python scripts/tools/fetch_danbooru_tags.py --out-dir data/danbooru/tags/raw --sleep 0.5

Environment:
    DANBOORU_USERNAME + DANBOORU_API_KEY  -> HTTP basic auth (higher rate limits for account holders).
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
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

BASE = "https://danbooru.donmai.us"
DEFAULT_UA = "sdx-danbooru-fetch/1.0 (tag export; respect rate limits)"
CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}


def _request(
    url: str,
    *,
    ua: str,
    auth_header: Optional[str],
    timeout: float = 120.0,
    retries: int = 4,
    sleep_base: float = 1.0,
) -> Any:
    last: Optional[Exception] = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "application/json"})
        if auth_header:
            req.add_header("Authorization", auth_header)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(sleep_base * (2**attempt))
                continue
            raise
        except OSError as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(sleep_base * (2**attempt))
                continue
            raise
    assert last is not None
    raise last


def _basic_auth_header() -> Optional[str]:
    user = (os.environ.get("DANBOORU_USERNAME") or "").strip()
    key = (os.environ.get("DANBOORU_API_KEY") or "").strip()
    if not user or not key:
        return None
    token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_pages(
    *,
    limit: int,
    max_pages: Optional[int],
    sleep_s: float,
    skip_deprecated: bool,
    ua: str,
    auth_header: Optional[str],
) -> DefaultDict[int, List[str]]:
    """
    Returns category_id -> lines ``name<TAB>post_count<TAB>deprecated``
    """
    buckets: DefaultDict[int, List[str]] = defaultdict(list)
    page = 1
    pages_done = 0
    while True:
        if max_pages is not None and pages_done >= max_pages:
            break
        q = urllib.parse.urlencode(
            {
                "limit": str(limit),
                "page": str(page),
                "search[order]": "id",
            }
        )
        url = f"{BASE}/tags.json?{q}"
        data = _request(url, ua=ua, auth_header=auth_header)
        if not isinstance(data, list) or not data:
            break
        for row in data:
            if not isinstance(row, dict):
                continue
            if skip_deprecated and row.get("is_deprecated"):
                continue
            cat = row.get("category")
            if cat not in CATEGORY_NAMES:
                continue
            name = row.get("name")
            if not name:
                continue
            pc = row.get("post_count", 0)
            dep = "1" if row.get("is_deprecated") else "0"
            buckets[int(cat)].append(f"{name}\t{pc}\t{dep}\n")
        pages_done += 1
        page += 1
        if len(data) < limit:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch all Danbooru tags into category files.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/danbooru/tags/raw"), help="Output directory")
    ap.add_argument("--limit", type=int, default=1000, help="Tags per page (max 1000)")
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (default: until empty)")
    ap.add_argument("--sleep", type=float, default=0.5, help="Delay between pages (seconds)")
    ap.add_argument("--skip-deprecated", action="store_true", help="Skip is_deprecated tags")
    ap.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent string")
    args = ap.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
