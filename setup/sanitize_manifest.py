#!/usr/bin/env python3
"""Drop manifest rows whose image files are missing or not trainable images."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrape.media_validate import TRAINABLE_IMAGE_EXTS, validate_trainable_image  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Remove broken/missing rows from a training manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--backup", action="store_true", help="Write manifest.bak before editing.")
    p.add_argument(
        "--verify-images",
        action="store_true",
        help="PIL-verify each image (slower; drops corrupt PNG/JPEG).",
    )
    args = p.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.data_root)
    if args.backup:
        shutil.copy(manifest, manifest.with_suffix(manifest.suffix + ".bak"))

    kept: list[str] = []
    removed = 0
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rel = d.get("image_path") or d.get("path") or d.get("image") or ""
            fp = root / str(rel)
            ok = fp.is_file() and fp.suffix.lower() in TRAINABLE_IMAGE_EXTS
            if ok and args.verify_images:
                ok = validate_trainable_image(fp)
            if ok:
                kept.append(line + "\n")
            else:
                removed += 1

    manifest.write_text("".join(kept), encoding="utf-8")
    print(f"kept {len(kept)} rows, removed {removed} -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
