#!/usr/bin/env python3
"""Harvest publicly shared prompts from Civitai (+ bundled PixAI / POV / edit recipes).

PixAI has no public browse-prompts API without an account key, so PixAI rows come
from their published docs/blog examples. Civitai rows come from GET /api/v1/images
with ``withMeta=true`` (prompts the uploader chose to share). Images are not saved.

    python -m scripts.tools harvest_community_prompts
    python scripts/tools/prompt/harvest_community_prompts.py --limit 80 --pages 2
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.prompt.community_recipes import (  # noqa: E402
    all_seed_recipes,
    bundled_corpus_path,
    is_blocked_prompt,
    recipe_from_mapping,
    recipes_to_jsonl,
)

_UA = "SDX-prompt-harvest/1.0 (research corpus; prompts only; +https://github.com)"


def _get_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def fetch_civitai_prompts(
    *,
    nsfw: str,
    limit: int,
    period: str = "Month",
    sort: str = "Most Reactions",
    pages: int = 1,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cursor = ""
    per = max(1, min(limit, 100))
    for _ in range(max(1, pages)):
        params: dict[str, str] = {
            "limit": str(per),
            "sort": sort,
            "period": period,
            "withMeta": "true",
            "nsfw": nsfw,
        }
        if cursor:
            params["cursor"] = cursor
        payload = _get_json(f"https://civitai.com/api/v1/images?{urllib.parse.urlencode(params)}")
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta") or {}
            if not isinstance(meta, dict):
                continue
            prompt = str(meta.get("prompt") or "").strip()
            if not prompt or is_blocked_prompt(prompt):
                continue
            negative = str(meta.get("negativePrompt") or "").strip()
            if is_blocked_prompt(negative):
                negative = ""
            iid = str(item.get("id") or "").strip()
            rows.append(
                {
                    "source": "civitai",
                    "source_id": iid,
                    "prompt": prompt[:2500],
                    "negative": negative[:800],
                    "note": f"nsfw={item.get('nsfwLevel', nsfw)} base={item.get('baseModel', '')}",
                }
            )
        cursor = str((payload.get("metadata") or {}).get("nextCursor") or "")
        if not cursor:
            break
    return rows


def harvest(
    *,
    limit: int = 80,
    pages: int = 2,
    include_mature: bool = True,
    include_x: bool = True,
    max_recipes: int = 280,
) -> list:
    recipes = all_seed_recipes()
    seen: set[str] = {r.prompt[:100].lower() for r in recipes}
    jobs: list[tuple[str, str, str]] = [
        ("None", "Most Reactions", "Month"),
        ("None", "Newest", "Week"),
        ("Soft", "Most Reactions", "Month"),
    ]
    if include_mature:
        jobs.append(("Mature", "Most Reactions", "Month"))
        jobs.append(("Mature", "Newest", "Week"))
    if include_x:
        jobs.append(("X", "Most Reactions", "Month"))
        jobs.append(("X", "Newest", "Week"))
    errors: list[str] = []
    for nsfw, sort, period in jobs:
        if len(recipes) >= max_recipes:
            break
        try:
            rows = fetch_civitai_prompts(
                nsfw=nsfw,
                limit=limit,
                period=period,
                sort=sort,
                pages=pages,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"civitai:{nsfw}:{sort}:{exc}")
            continue
        for row in rows:
            rec = recipe_from_mapping(row)
            if rec is None:
                continue
            key = rec.prompt[:100].lower()
            if key in seen:
                continue
            seen.add(key)
            recipes.append(rec)
            if len(recipes) >= max_recipes:
                break
    if errors:
        print("Harvest warnings:", "; ".join(errors), file=sys.stderr)
    return recipes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Harvest public Civitai/PixAI prompts into the research corpus.")
    p.add_argument("--limit", type=int, default=80, help="Per-request Civitai page size (max 100).")
    p.add_argument("--pages", type=int, default=2, help="Cursor pages per (nsfw, sort) job.")
    p.add_argument("--out", default="", help="JSONL path (default: config/defaults/community_prompt_recipes.jsonl).")
    p.add_argument("--no-mature", action="store_true", help="Skip Civitai Mature bucket.")
    p.add_argument("--no-x", action="store_true", help="Skip Civitai X (explicit) bucket.")
    p.add_argument("--offline", action="store_true", help="Write bundled teaching prompts only (no network).")
    args = p.parse_args(argv)

    out = Path(args.out) if str(args.out).strip() else bundled_corpus_path()
    if args.offline:
        recipes = all_seed_recipes()
    else:
        recipes = harvest(
            limit=int(args.limit),
            pages=int(args.pages),
            include_mature=not args.no_mature,
            include_x=not args.no_x,
        )
    n = recipes_to_jsonl(recipes, out)
    lanes = {"simple": 0, "intermediate": 0, "hard": 0}
    for r in recipes:
        lanes[r.lane] = lanes.get(r.lane, 0) + 1
    print(f"Wrote {n} recipes -> {out}")
    print(f"lanes simple={lanes['simple']} intermediate={lanes['intermediate']} hard={lanes['hard']}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
