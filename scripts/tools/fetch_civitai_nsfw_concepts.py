#!/usr/bin/env python3
"""
Fetch Civitai model names + trainedWords (trigger words) via the public REST API.

The site search URL
  https://civitai.com/search/models?category=concept&baseModel=Illustrious&baseModel=NoobAI&sortBy=models_v9&query=nsfw
is JavaScript-driven; this script uses the documented API instead:
  https://developer.civitai.com/docs/api/public-rest

The API does not expose the web "Concept" category filter; this tool approximates the
intent with ``query=nsfw`` + optional type filters, then keeps rows whose *modelVersions*
use base models Illustrious and/or NoobAI.

Pagination uses cursor links from ``metadata.nextPage`` (``page=`` returns 400).

Usage:
  python -m scripts.tools fetch_civitai_nsfw_concepts --out data/civitai/nsfw_illustrious_noobai_models.csv
  python -m scripts.tools fetch_civitai_nsfw_concepts --max-batches 40 --sleep 0.35
  python -m scripts.tools fetch_civitai_nsfw_concepts --preset extended --max-batches-per-query 8 --out data/civitai/nsfw_illustrious_noobai_models.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

API_MODELS = "https://civitai.com/api/v1/models"
DEFAULT_UA = "Mozilla/5.0 (compatible; sdx-fetch/1.0; +https://github.com/)"
TARGET_BASES = frozenset({"Illustrious", "NoobAI"})

# Approximates the user's Civitai *site* searches (category filters are not in the public API;
# we use ``query=`` only). Deduped. See ``data/civitai/SEARCHES.md``.
# Includes: clothing/concept/character/style themed searches + global sortBy=models_v9 style queries.
EXTENDED_SEARCH_QUERIES: List[str] = [
    "nsfw",
    "",  # approx. site "browse" with no text query (e.g. clothing category-only URL)
    "clothing",
    "sex",
    "hentai",
    "succubus",
    "porn",
    "game",
    "anime",
    "furry",
    "cartoon",
    "artist",
    "3d",
    "realistic",
    "waifu",
    "cum",
    "manga",
    "position",
    "pov",
    "intercrural",
    "thigh sex",
    "standing sex",
    "concept",
    "clothes",
    "sex toy",
    "slime",
    "sfm",
    "passionate",
    "paizuri",
    "fellatio",
    "2d",
]


def _request_json(
    url: str,
    *,
    token: Optional[str] = None,
    timeout: float = 120.0,
    retries: int = 3,
    sleep_s: float = 0.25,
) -> Dict[str, Any]:
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={token}"
    last_err: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(sleep_s * (2**attempt))
                continue
            raise
        except OSError as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(sleep_s * (2**attempt))
                continue
            raise
    assert last_err is not None
    raise last_err


def _split_trained_words(raw: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for chunk in raw:
        if chunk is None:
            continue
        for part in str(chunk).split(","):
            t = part.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


def _merge_rows(
    a: Tuple[int, str, str, str, str],
    b: Tuple[int, str, str, str, str],
) -> Tuple[int, str, str, str, str]:
    """Same model id: union bases and triggers (stable order, first row wins for name/type)."""
    _id, name, mtype, bases_a, trig_a = a
    _id_b, _name_b, _type_b, bases_b, trig_b = b
    bases = sorted(
        (set((bases_a or "").split("|")) | set((bases_b or "").split("|"))) - {""},
    )
    seen_t: Set[str] = set()
    out_trig: List[str] = []
    for raw in (trig_a, trig_b):
        for t in (raw or "").split("|"):
            x = t.strip()
            if not x:
                continue
            k = x.lower()
            if k in seen_t:
                continue
            seen_t.add(k)
            out_trig.append(x)
    return (_id, name, mtype, "|".join(bases), "|".join(out_trig))


def _model_row(item: Dict[str, Any]) -> Optional[Tuple[int, str, str, str, str]]:
    """Return (id, name, type, bases_joined, triggers_joined) or None if no target base."""
    versions = item.get("modelVersions") or []
    bases: Set[str] = set()
    triggers: List[str] = []
    trig_seen: Set[str] = set()
    for ver in versions:
        bm = ver.get("baseModel") or ""
        if bm not in TARGET_BASES:
            continue
        bases.add(bm)
        for t in _split_trained_words(ver.get("trainedWords") or []):
            if t not in trig_seen:
                trig_seen.add(t)
                triggers.append(t)
    if not bases:
        return None
    mid = int(item["id"])
    name = (item.get("name") or "").replace("\r\n", " ").replace("\n", " ").strip()
    mtype = (item.get("type") or "").strip()
    return (mid, name, mtype, "|".join(sorted(bases)), "|".join(triggers))


def fetch_all(
    *,
    query: str = "nsfw",
    nsfw: bool = True,
    limit: int = 100,
    max_batches: int = 25,
    types: Optional[List[str]] = None,
    token: Optional[str] = None,
    sleep_s: float = 0.25,
) -> List[Tuple[int, str, str, str, str]]:
    params: List[str] = [f"limit={int(limit)}"]
    q = (query or "").strip()
    if q:
        params.insert(0, f"query={urllib.parse.quote(q)}")
    if nsfw:
        params.append("nsfw=true")
    if types:
        for t in types:
            params.append("types=" + urllib.parse.quote(t))
    url = API_MODELS + "?" + "&".join(params)

    rows: List[Tuple[int, str, str, str, str]] = []
    seen_ids: Set[int] = set()
    batches = 0
    while url and batches < max_batches:
        batches += 1
        try:
            data = _request_json(url, token=token, sleep_s=max(0.1, sleep_s))
        except urllib.error.HTTPError as e:
            print(f"HTTP error {e.code} for {url[:120]}...", file=sys.stderr)
            raise
        for item in data.get("items") or []:
            row = _model_row(item)
            if row is None:
                continue
            rid = row[0]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rows.append(row)
        meta = data.get("metadata") or {}
        url = meta.get("nextPage") or ""
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows


def fetch_merged_queries(
    queries: List[str],
    *,
    nsfw: bool = True,
    limit: int = 100,
    max_batches_per_query: int = 8,
    types: Optional[List[str]] = None,
    token: Optional[str] = None,
    sleep_s: float = 0.25,
) -> List[Tuple[int, str, str, str, str]]:
    """
    Run several API searches and merge by model id (union triggers + bases).

    Empty strings mean one pass **without** a ``query=`` API parameter (broad browse).
    Whitespace-only strings are skipped.
    """
    merged: Dict[int, Tuple[int, str, str, str, str]] = {}
    lim = min(100, max(1, limit))
    mb = max(1, max_batches_per_query)
    for raw_q in queries:
        if raw_q is None:
            continue
        s = str(raw_q)
        if s == "":
            q = ""  # explicit placeholder in query list = API call without query=
        else:
            q = s.strip()
            if not q:
                continue
        label = "(no query / browse)" if q == "" else repr(q)
        print(f"--- query {label} ---", file=sys.stderr)
        try:
            part = fetch_all(
                query=q,
                nsfw=nsfw,
                limit=lim,
                max_batches=mb,
                types=types,
                token=token,
                sleep_s=sleep_s,
            )
        except (urllib.error.HTTPError, OSError) as e:
            print(f"SKIP query {label}: {e}", file=sys.stderr)
            continue
        for row in part:
            rid = row[0]
            if rid not in merged:
                merged[rid] = row
            else:
                merged[rid] = _merge_rows(merged[rid], row)
    return sorted(merged.values(), key=lambda r: r[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Civitai NSFW Illustrious/NoobAI model triggers via API.")
    parser.add_argument(
        "--out",
        default="data/civitai/nsfw_illustrious_noobai_models.csv",
        help="Output CSV path (id,name,type,bases,triggers).",
    )
    parser.add_argument("--query", default="nsfw", help="Single API search query (default: nsfw). Ignored if --preset extended.")
    parser.add_argument(
        "--preset",
        choices=["none", "extended"],
        default="none",
        help="extended = merge many queries approximating saved Civitai site searches (Illustrious/NoobAI filter still applies).",
    )
    parser.add_argument(
        "--extra-queries",
        default="",
        help="Comma-separated extra queries to merge after --preset (or after single --query).",
    )
    parser.add_argument("--limit", type=int, default=100, help="Batch size (max 100).")
    parser.add_argument("--max-batches", type=int, default=25, help="Max API pages per query (single-query mode).")
    parser.add_argument(
        "--max-batches-per-query",
        type=int,
        default=8,
        help="Max API pages per query when using --preset extended.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between requests.")
    parser.add_argument("--token", default=None, help="Optional Civitai API token (query param).")
    parser.add_argument(
        "--types",
        default="",
        help="Comma-separated API model types to filter (e.g. LORA,TextualInversion). Empty = all.",
    )
    parser.add_argument(
        "--no-nsfw-flag",
        action="store_true",
        help="Omit nsfw=true on the API request (broader, more SFW results; usually not needed).",
    )
    args = parser.parse_args()
    types_list = [t.strip() for t in args.types.split(",") if t.strip()] or None
    nsfw_api = not args.no_nsfw_flag
    extras = [x.strip() for x in args.extra_queries.split(",") if x.strip()]

    if args.preset == "extended":
        queries = list(EXTENDED_SEARCH_QUERIES) + extras
        rows = fetch_merged_queries(
            queries,
            nsfw=nsfw_api,
            limit=min(100, max(1, args.limit)),
            max_batches_per_query=max(1, args.max_batches_per_query),
            types=types_list,
            token=args.token,
            sleep_s=max(0.0, args.sleep),
        )
    else:
        qlist = [args.query] + extras
        if len(qlist) == 1 and not extras:
            rows = fetch_all(
                query=args.query,
                nsfw=nsfw_api,
                limit=min(100, max(1, args.limit)),
                max_batches=max(1, args.max_batches),
                types=types_list,
                token=args.token,
                sleep_s=max(0.0, args.sleep),
            )
        else:
            rows = fetch_merged_queries(
                qlist,
                nsfw=nsfw_api,
                limit=min(100, max(1, args.limit)),
                max_batches_per_query=max(1, args.max_batches),
                types=types_list,
                token=args.token,
                sleep_s=max(0.0, args.sleep),
            )
    out_path = args.out
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "type", "bases", "triggers"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} models to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
