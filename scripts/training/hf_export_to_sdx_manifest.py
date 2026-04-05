#!/usr/bin/env python3
"""
Export a **Hugging Face Dataset** to SDX training format: images on disk + JSONL manifest.

Designed for tag-style captions (e.g. Danbooru: comma-separated tags). Works when the dataset
provides an **image** column (PIL / HF Image / bytes). If your dataset is **metadata-only**
(tags + post id, no pixels), you must pair images separately—see docs/DANBOORU_HF.md.

Usage (install: ``pip install datasets``):

    python scripts/training/hf_export_to_sdx_manifest.py \\
        --dataset YOUR_ORG/danbooru-dataset \\
        --split train \\
        --image-field image \\
        --caption-field tag_string \\
        --out-dir data/hf_danbooru \\
        --max-samples 5000

Then train:

    python train.py --manifest-jsonl data/hf_danbooru/manifest.jsonl --data-path data/hf_danbooru \\
        --model DiT-B/2-Text --image-size 256 --results-dir results/danbooru

Use ``--streaming`` for huge datasets; pair with ``--max-samples`` to cap disk use.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_pil(image_val: Any):
    """HF Image feature, PIL, numpy, or dict with bytes."""
    if image_val is None:
        return None
    if hasattr(image_val, "save"):
        return image_val
    if isinstance(image_val, dict):
        b = image_val.get("bytes")
        if b is not None:
            from PIL import Image

            return Image.open(io.BytesIO(b)).convert("RGB")
        p = image_val.get("path")
        if p:
            from PIL import Image

            return Image.open(p).convert("RGB")
    try:
        import numpy as np

        if isinstance(image_val, np.ndarray):
            from PIL import Image

            arr = image_val
            if arr.ndim == 2:
                return Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
            return Image.fromarray(arr.astype("uint8"), mode="RGB")
    except Exception:
        pass
    return None


def _caption_from_row(row: Dict[str, Any], caption_field: str, tag_join: str) -> str:
    v = row.get(caption_field)
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return tag_join.join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def main() -> int:
    p = argparse.ArgumentParser(description="HF Dataset -> SDX JSONL + image files.")
    p.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. org/danbooru-export")
    p.add_argument("--config", type=str, default=None, help="Optional dataset config name (second arg to load_dataset).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", type=str, default=None, help="Git revision / branch / commit.")
    p.add_argument("--image-field", type=str, default="image", help="Column with image (PIL/bytes).")
    p.add_argument("--caption-field", type=str, default="tag_string", help="Column for caption/tags.")
    p.add_argument(
        "--caption-tag-join",
        type=str,
        default=", ",
        help="If caption field is a list of tags, join with this string (default ', ').",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output root; images/ + manifest.jsonl")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N rows (0 = no limit).")
    p.add_argument("--streaming", action="store_true", help="IterableDataset mode (recommended for huge data).")
    p.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed when streaming (buffered shuffle).")
    p.add_argument("--start-index", type=int, default=0, help="Skip first N rows (non-streaming only).")
    p.add_argument("--list-columns", action="store_true", help="Print first row keys and exit.")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest_name

    ld_kw: Dict[str, Any] = {}
    if args.revision:
        ld_kw["revision"] = args.revision
    ds_id = args.dataset
    cfg = args.config

    def _load_streaming():
        if cfg is not None:
            return load_dataset(ds_id, cfg, split=args.split, streaming=True, **ld_kw)
        return load_dataset(ds_id, split=args.split, streaming=True, **ld_kw)

    def _load_full():
        if cfg is not None:
            return load_dataset(ds_id, cfg, **ld_kw)
        return load_dataset(ds_id, **ld_kw)

    if args.streaming:
        split_ds = _load_streaming()
        if args.shuffle_seed is not None:
            split_ds = split_ds.shuffle(seed=int(args.shuffle_seed), buffer_size=10_000)

        if args.list_columns:
            it = iter(split_ds)
            row = next(it)
            print("Columns:", list(row.keys()) if isinstance(row, dict) else row)
            return 0

        iterator = split_ds
    else:
        raw = _load_full()
        split_ds = raw[args.split] if isinstance(raw, dict) else raw
        if args.start_index:
            split_ds = split_ds.select(range(int(args.start_index), len(split_ds)))

        if args.list_columns:
            print("Columns:", split_ds.column_names)
            row = split_ds[0]
            print("First row keys:", list(row.keys()) if isinstance(row, dict) else type(row))
            return 0

        iterator = split_ds

    n_written = 0

    def rows():
        if args.streaming:
            for row in iterator:
                yield row
        else:
            for i in range(len(iterator)):
                yield iterator[i]

    with manifest_path.open("w", encoding="utf-8") as mf:
        for row in rows():
            if args.max_samples and n_written >= int(args.max_samples):
                break
            if not isinstance(row, dict):
                row = dict(row)

            cap = _caption_from_row(row, args.caption_field, args.caption_tag_join)
            if not cap:
                continue

            pil = _to_pil(row.get(args.image_field))
            if pil is None:
                continue

            stem = f"{n_written:08d}"
            ipath = img_dir / f"{stem}.png"
            pil.save(ipath)

            rec = {
                "image_path": str(ipath.resolve()),
                "caption": cap,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 500 == 0:
                print(f"  wrote {n_written} samples …", flush=True)

    print(f"Done: {n_written} rows -> {manifest_path}")
    print(f"Train with: python train.py --manifest-jsonl {manifest_path} --data-path {out_dir} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
