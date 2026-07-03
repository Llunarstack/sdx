#!/usr/bin/env python3
"""Merge per-site scrape manifests into one JSONL for multi-site training.

Each site's ``manifest.jsonl`` uses paths relative to that site's folder. This
rewrites ``image_path`` to ``<site>/images/<file>`` so a single ``--data-path``
root can serve every site.

    python setup/merge_manifests.py --data-root /workspace/data --out /workspace/data/combined/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.sites import ADAPTERS, DEFAULT_SCRAPE_SITES  # noqa: E402


def merge_manifests(data_root: Path, out_path: Path, sites: list[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_md5: set[str] = set()
    written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for site in sites:
            manifest = data_root / site / "manifest.jsonl"
            if not manifest.is_file():
                print(f"[skip] {site}: no {manifest}")
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                md5 = row.get("md5", "")
                if md5 and md5 in seen_md5:
                    continue
                if md5:
                    seen_md5.add(md5)
                rel = row.get("image_path", "")
                if rel and not str(rel).startswith(f"{site}/"):
                    row["image_path"] = f"{site}/{rel}".replace("\\", "/")
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"[ok] {site}: merged into {out_path.name}")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge booru site manifests (dedupe by md5).")
    p.add_argument("--data-root", required=True, help="Base dir containing per-site folders.")
    p.add_argument("--out", required=True, help="Output combined manifest.jsonl path.")
    p.add_argument("--sites", nargs="*", default=list(DEFAULT_SCRAPE_SITES), help="Sites to include.")
    args = p.parse_args(argv)
    n = merge_manifests(Path(args.data_root), Path(args.out), list(args.sites))
    print(f"Wrote {n:,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
