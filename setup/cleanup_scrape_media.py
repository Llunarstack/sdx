#!/usr/bin/env python3
"""Remove junk media from scraped booru folders before a fresh crawl or train.

Deletes:
  - ``*.part`` / ``*.tmp`` incomplete downloads
  - ``zip/swf/html`` and other blocked extensions
  - raw ``gif/mp4/webm/...`` when frame JPEGs already exist (or --drop-raw-media)
  - corrupt images that fail PIL validation

Optionally rewrites per-site ``manifest.jsonl`` to drop rows pointing at removed files.

    python setup/cleanup_scrape_media.py --data-root /workspace/data --sites danbooru rule34xxx
    python setup/cleanup_scrape_media.py --data-root /workspace/data --rewrite-manifests --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.media_validate import (  # noqa: E402
    BLOCKED_DOWNLOAD_EXTS,
    SPLITTABLE_SOURCE_EXTS,
    TRAINABLE_IMAGE_EXTS,
    validate_trainable_image,
)
from scripts.scrape.sites import DEFAULT_SCRAPE_SITES  # noqa: E402


def _stem_parent_md5(path: Path) -> str | None:
    name = path.stem
    if "_f" in name:
        return name.rsplit("_f", 1)[0]
    return name


def cleanup_images_dir(
    images_dir: Path,
    *,
    drop_raw_media: bool,
    drop_invalid: bool,
) -> dict[str, int]:
    counts = {
        "part": 0,
        "blocked_ext": 0,
        "raw_with_frames": 0,
        "raw_orphan": 0,
        "invalid_image": 0,
        "kept": 0,
    }
    if not images_dir.is_dir():
        return counts

    frame_parents: set[str] = set()
    for p in images_dir.glob("*_f*.jpg"):
        parent = _stem_parent_md5(p)
        if parent:
            frame_parents.add(parent)

    for path in list(images_dir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in {".part", ".tmp", ".crdownload"} or path.name.endswith(".part"):
            path.unlink(missing_ok=True)
            counts["part"] += 1
            continue
        if ext in BLOCKED_DOWNLOAD_EXTS:
            path.unlink(missing_ok=True)
            counts["blocked_ext"] += 1
            continue
        if ext in SPLITTABLE_SOURCE_EXTS:
            parent = path.stem
            if parent in frame_parents or drop_raw_media:
                path.unlink(missing_ok=True)
                if parent in frame_parents:
                    counts["raw_with_frames"] += 1
                else:
                    counts["raw_orphan"] += 1
            continue
        if ext in TRAINABLE_IMAGE_EXTS:
            if drop_invalid and not validate_trainable_image(path):
                path.unlink(missing_ok=True)
                counts["invalid_image"] += 1
            else:
                counts["kept"] += 1
            continue
        path.unlink(missing_ok=True)
        counts["blocked_ext"] += 1
    return counts


def rewrite_manifest(manifest: Path, site_root: Path, *, backup: bool) -> tuple[int, int]:
    if not manifest.is_file():
        return 0, 0
    if backup:
        shutil.copy(manifest, manifest.with_suffix(manifest.suffix + ".bak"))
    kept: list[str] = []
    removed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rel = row.get("image_path") or row.get("path") or ""
        fp = site_root / str(rel)
        if fp.is_file() and fp.suffix.lower() in TRAINABLE_IMAGE_EXTS:
            if validate_trainable_image(fp):
                kept.append(line)
                continue
        removed += 1
    manifest.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept), removed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean scraped booru media folders.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--sites", nargs="*", default=list(DEFAULT_SCRAPE_SITES))
    p.add_argument(
        "--drop-raw-media",
        action="store_true",
        help="Delete all gif/mp4/webm even when frame JPEGs are missing.",
    )
    p.add_argument("--rewrite-manifests", action="store_true")
    p.add_argument("--backup", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.data_root)
    totals = {k: 0 for k in ("part", "blocked_ext", "raw_with_frames", "raw_orphan", "invalid_image", "kept")}
    for site in args.sites:
        site_root = root / site
        images = site_root / "images"
        stats = cleanup_images_dir(
            images,
            drop_raw_media=bool(args.drop_raw_media),
            drop_invalid=True,
        )
        print(f"[{site}] images: {stats}")
        for k, v in stats.items():
            totals[k] += v
        if args.rewrite_manifests:
            manifest = site_root / "manifest.jsonl"
            kept, removed = rewrite_manifest(manifest, site_root, backup=args.backup)
            print(f"[{site}] manifest: kept {kept}, removed {removed}")
    print(f"total: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
