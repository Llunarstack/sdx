#!/usr/bin/env python3
"""Run WD EVA02 tagger over a manifest and merge supplementary tags into captions.

Identity tags (character, artist, copyright) stay from the booru API; WD adds
hair, pose, clothing, scene, etc. Run **before** VLM enrichment.

Minor screen (always on): a row is dropped when its WD tags at ``SCREEN_THRESHOLD``
or its manifest text trip ``utils.safety.minor_guard``'s training rule (any youth
term, any rating). It catches images whose booru tags omit youth tags (Thiel,
Stanford Internet Observatory, 2023, on CSAM in LAION-5B).

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

from data.manifest_utils import manifest_row_text, resolve_image_path  # noqa: E402
from utils.caption.wd_tagger import WDTagger, merge_wd_tags_into_row  # noqa: E402
from utils.safety.minor_guard import reasons as minor_guard_reasons  # noqa: E402

# Below the 0.35 caption threshold: a missed youth tag costs more than a dropped adult row.
SCREEN_THRESHOLD = 0.2


def _resolve_image(data_root: Path, row: dict) -> Path | None:
    rel = str(row.get("image_path") or row.get("path") or "").strip()
    site = str(row.get("site") or "").strip()
    found = resolve_image_path(rel, data_root)
    if found is None and rel and site:
        cand = data_root / site / "images" / Path(rel).name
        found = cand if cand.is_file() else None
    return found


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
    youth_vocab = [n for n in tagger.tag_names if minor_guard_reasons(n, training=True)]
    if not youth_vocab:
        raise SystemExit(
            f"{tagger.model_dir / 'selected_tags.csv'} has no youth tags; the minor screen would be a no-op."
        )
    print(f"WD minor screen: {len(youth_vocab)} youth tags in the vocabulary ({', '.join(youth_vocab[:8])}, ...)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = 0
    errors = 0
    dropped = 0

    with tmp.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            done = skip_complete and "wd_tagger" in [str(s) for s in (row.get("tag_sources") or [])]
            img = None if done else _resolve_image(data_root, row)
            if not done and img is None:
                errors += 1
            if img is not None:
                try:
                    tags = tagger.scores(img)
                except Exception as exc:
                    # An image the screen cannot see is not written (fail closed).
                    print(f"  row {i}: WD tag failed, row dropped ({exc})", file=sys.stderr)
                    errors += 1
                    continue
                if minor_guard_reasons(", ".join(t.name for t in tags if t.score >= SCREEN_THRESHOLD), training=True):
                    dropped += 1
                    continue
                row = merge_wd_tags_into_row(row, [t.name for t in tagger.select(tags)])
            if minor_guard_reasons(manifest_row_text(row), training=True):
                dropped += 1
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if i % 500 == 0:
                print(f"  tagged {i}/{len(rows)}...", flush=True)

    tmp.replace(out)
    print(f"WD-tagged {written} rows -> {out} ({errors} errors, {dropped} dropped by the minor screen)")
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
