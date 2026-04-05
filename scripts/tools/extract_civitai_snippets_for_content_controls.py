#!/usr/bin/env python3
"""
One-off / dev helper: mine short snippets from the Civitai CSV for pasting into content_controls.

Usage:
  python scripts/tools/extract_civitai_snippets_for_content_controls.py > snippets.txt
"""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "data" / "civitai" / "nsfw_illustrious_noobai_models.csv"


def _load_civitai_hot_lowercase() -> set[str]:
    """Load CIVITAI_HOT_TAGS without importing content_controls (avoids circular import)."""
    path = REPO / "utils" / "prompt" / "civitai_vocab.py"
    spec = importlib.util.spec_from_file_location("civitai_vocab_standalone", path)
    if spec is None or spec.loader is None:
        return set()
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tags = getattr(mod, "CIVITAI_HOT_TAGS", []) or []
    return {str(x).lower() for x in tags}


def _load_existing_lowercase() -> set[str]:
    existing: set[str] = set()
    existing.update(_load_civitai_hot_lowercase())
    for x in (
        "score_9",
        "score_8_up",
        "score_7_up",
        "masterpiece",
        "best quality",
        "amazing quality",
        "newest",
        "absurdres",
        "highres",
    ):
        existing.add(x.lower())
    return existing


STOP_NAME = frozenset(
    {
        "lora",
        "locon",
        "lycoris",
        "checkpoint",
        "illustrious",
        "noobai",
        "noob",
        "pony",
        "sdxl",
        "sd",
        "xl",
        "nsfw",
        "support",
        "commission",
        "version",
        "mix",
        "concept",
        "artstyle",
        "by",
        "1.5",
        "2.0",
        "2.1",
        "v1",
        "v2",
        "v3",
        "v4",
        "the",
        "and",
        "or",
        "for",
        "model",
        "trained",
        "embedding",
        "wan",
        "i2v",
        "workflow",
        "t2i",
        "pack",
        "ultimate",
        "edition",
        "beta",
        "alpha",
        "goofy",
        "ai",
        "character",
        "characters",
        "style",
        "merge",
        "merged",
        "official",
        "unofficial",
        "experimental",
        "deprecated",
    }
)


def clean_trigger_chunk(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\([^)]*\)\s*\|?\s*", "", s)
    s = re.sub(r"\([^)]*:[\d.]+\)", "", s)
    return s.strip(" |")


def split_triggers(raw: str):
    if not raw:
        return
    for part in raw.split("|"):
        part = clean_trigger_chunk(part)
        if not part or len(part) > 45:
            continue
        if "," in part and len(part) > 35:
            for bit in part.split(","):
                b = bit.strip()
                if 2 <= len(b) <= 40:
                    yield b
        elif 2 <= len(part) <= 40:
            yield part


def name_snippets(name: str):
    if not name:
        return
    n = re.sub(r"\[[^\]]*\]", " ", name)
    n = re.sub(r"\([^)]{0,80}\)", " ", n)
    n = n.replace('"', "").replace("\u201c", "").replace("\u201d", "")
    for sep in ["|", "/", "–", "—", "-", ":"]:
        n = n.replace(sep, " ")
    n = re.sub(r"\s+", " ", n).strip()
    for chunk in re.split(r"\s*[-–—]\s*", n):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 3 or len(chunk) > 42:
            continue
        words = chunk.split()
        if len(words) > 6:
            continue
        lw = {w.lower().strip(".,;") for w in words}
        if lw <= STOP_NAME:
            continue
        if chunk.lower() in STOP_NAME:
            continue
        yield chunk


def main() -> int:
    existing = _load_existing_lowercase()
    ctr: Counter[str] = Counter()
    if not CSV_PATH.is_file():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        return 1

    for row in csv.DictReader(CSV_PATH.open(encoding="utf-8")):
        for t in split_triggers(row.get("triggers") or ""):
            k = t.lower()
            if k in existing:
                continue
            if re.match(r"^[\d\s\-.]+$", t):
                continue
            ctr[t] += 1
        for sn in name_snippets(row.get("name") or ""):
            k = sn.lower()
            if k in existing:
                continue
            ctr[sn] += 1

    out: list[str] = []
    for token, c in ctr.most_common(500):
        if c < 2 and len(token.split()) >= 4:
            continue
        if c < 2 and any(x in token.lower() for x in ["http", "www", ".com", "discord"]):
            continue
        kl = token.lower()
        if kl in existing:
            continue
        existing.add(kl)
        out.append(token)
        if len(out) >= 96:
            break

    sys.stderr.write(f"wrote {len(out)} snippet lines (deduped vs CIVITAI_HOT_TAGS + score tags)\n")
    for x in out:
        print(repr(x) + ",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
