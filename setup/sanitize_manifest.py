#!/usr/bin/env python3
"""Drop manifest rows whose image files are missing or not trainable images."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TRAINABLE = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})


def main() -> int:
    p = argparse.ArgumentParser(description="Remove broken/missing rows from a training manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--backup", action="store_true", help="Write manifest.bak before editing.")
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
            if fp.is_file() and fp.suffix.lower() in TRAINABLE:
                kept.append(line + "\n")
            else:
                removed += 1

    manifest.write_text("".join(kept), encoding="utf-8")
    print(f"kept {len(kept)} rows, removed {removed} -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
