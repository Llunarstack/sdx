#!/usr/bin/env python3
"""Build ``artist_index.json`` from scraped booru manifests.

Every artist tag seen in danbooru / e621 downloads is indexed so ``@AnyName`` in
prompts resolves to the spelling the model trained on.

    python setup/build_artist_index.py --data-root /workspace/data
    python setup/build_artist_index.py --manifest a/manifest.jsonl --manifest b/manifest.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.sites import ADAPTERS  # noqa: E402
from utils.prompt.artist_registry import build_from_manifests  # noqa: E402


def _discover_manifests(data_root: Path, sites: list[str]) -> list[Path]:
    paths: list[Path] = []
    for site in sites:
        p = data_root / site / "manifest.jsonl"
        if p.is_file():
            paths.append(p)
    combined = data_root / "combined" / "manifest.jsonl"
    if combined.is_file():
        paths.append(combined)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build artist alias index from scrape manifests.")
    p.add_argument("--data-root", default=None, help="Scan <root>/<site>/manifest.jsonl for all sites.")
    p.add_argument("--manifest", action="append", default=[], help="Explicit manifest.jsonl path (repeatable).")
    p.add_argument("--out", default=None, help="Output JSON (default: <data-root>/artist_index.json).")
    p.add_argument("--sites", nargs="*", default=sorted(ADAPTERS))
    args = p.parse_args(argv)

    manifests: list[Path] = [Path(m) for m in args.manifest]
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None:
        manifests.extend(_discover_manifests(data_root, list(args.sites)))
    manifests = list(dict.fromkeys(manifests))
    if not manifests:
        print("No manifests found. Pass --data-root or --manifest.", file=sys.stderr)
        return 2

    reg = build_from_manifests(manifests)
    out = (
        Path(args.out)
        if args.out
        else (data_root / "artist_index.json" if data_root else REPO_ROOT / "data" / "artist_index.json")
    )
    reg.save(out)
    print(f"Indexed {len(reg):,} artists from {len(manifests)} manifest(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
