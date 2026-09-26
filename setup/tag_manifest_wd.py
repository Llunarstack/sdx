#!/usr/bin/env python3
"""Run WD EVA02 tagger over a manifest and merge supplementary tags into captions.

Identity tags (character, artist, copyright) stay from the booru API; WD adds
hair, pose, clothing, scene, etc. Run **before** VLM enrichment.

    python setup/tag_manifest_wd.py \\
        --manifest /workspace/data/combined/manifest.jsonl \\
        --data-root /workspace/data \\
        --out /workspace/data/tagged/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.caption.wd_tagger import WDTagger, merge_wd_tags_into_row  # noqa: E402


def _resolve_image(data_root: Path, row: dict) -> Path | None:
    rel = str(row.get("image_path") or row.get("path") or "").strip()
    if not rel:
        return None
    p = Path(rel)
    if p.is_file():
        return p
    cand = data_root / rel
    if cand.is_file():
        return cand
    site = str(row.get("site") or "").strip()
    if site:
        cand = data_root / site / "images" / p.name
        if cand.is_file():
            return cand
    return None


def tag_manifest(
    manifest: Path,
    data_root: Path,
    out: Path,
    *,
    model_dir: Path | None = None,
    threshold: float = 0.35,
    max_rows: int = 0,
    skip_complete: bool = True,
) -> int:
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_rows > 0:
        rows = rows[:max_rows]

    tagger = WDTagger(model_dir, threshold=threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0
    errors = 0

    with tmp.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            if skip_complete and "wd_tagger" in [str(s) for s in (row.get("tag_sources") or [])]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                continue
            img = _resolve_image(data_root, row)
            if img is None:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                errors += 1
                continue
            try:
                names = tagger.predict_names(img)
                row = merge_wd_tags_into_row(row, names)
            except Exception as exc:
                print(f"  row {i}: WD tag failed ({exc})", file=sys.stderr)
                errors += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if i % 500 == 0:
                print(f"  tagged {i}/{len(rows)}...", flush=True)

    tmp.replace(out)
    print(f"WD-tagged {written} rows -> {out} ({errors} skipped/errors)")
    return 0 if written else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WD tagger manifest enrichment.")
    p.add_argument("--manifest", required=True, help="Input manifest.jsonl")
    p.add_argument("--data-root", required=True, help="Dataset root for image paths")
    p.add_argument("--out", required=True, help="Output manifest.jsonl")
    p.add_argument("--model-dir", default=None, help="WD-EVA02-Large-Tagger folder")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Re-tag rows that already have wd_tagger")
    args = p.parse_args(argv)

    return tag_manifest(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.out),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        threshold=args.threshold,
        max_rows=args.max_rows,
        skip_complete=not args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
