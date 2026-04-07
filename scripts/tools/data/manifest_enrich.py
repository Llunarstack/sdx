#!/usr/bin/env python3
"""
Add cheap curation signals to a training manifest JSONL:

- ``clip_sim``: CLIP image–text similarity (transformers) vs caption
- ``aesthetic_proxy``: heuristic [0,1] blend (exposure, tiling, dynamic range) — no extra weights
- ``dedupe_phash``: optional perceptual hash for near-dup filtering (imagehash)

Paths are resolved like ``data_quality``: absolute paths, ``--image-root``, or relative to the manifest file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_image(path_str: str, *, image_root: Optional[Path], manifest_dir: Path) -> Optional[Path]:
    p = Path(path_str)
    if p.is_file():
        return p
    if image_root:
        q = (image_root / path_str).resolve()
        if q.is_file():
            return q
    q = (manifest_dir / path_str).resolve()
    if q.is_file():
        return q
    return None


def _load_rgb_uint8(path: Path) -> Optional[Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(path).convert("RGB")
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def _perceptual_hash_hex(path: Path, size: int = 8) -> str:
    try:
        import imagehash
        from PIL import Image

        return str(imagehash.phash(Image.open(path), hash_size=size))
    except Exception:
        return ""


def _aesthetic_proxy(rgb) -> float:
    from utils.quality.test_time_pick import score_aesthetic_proxy

    return float(score_aesthetic_proxy(rgb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=str, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Output JSONL (default: input stem + _enriched.jsonl)")
    ap.add_argument("--image-root", type=str, default="", help="Base dir for relative image paths")
    ap.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP id (same family as sample.py / test_time_pick)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after N valid rows (0 = all)")
    ap.add_argument("--skip-clip", action="store_true", help="Only add aesthetic_proxy / phash (no transformers)")
    ap.add_argument("--with-phash", action="store_true", help="Add dedupe_phash column (needs imagehash + PIL)")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else inp.parent / f"{inp.stem}_enriched.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).resolve() if args.image_root else None
    manifest_dir = inp.parent

    device = args.device
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

    score_clip_similarity = None
    if not args.skip_clip:
        from utils.quality.test_time_pick import score_clip_similarity as _sc

        score_clip_similarity = _sc

    n_ok = n_skip = n_bad_path = 0
    with inp.open(encoding="utf-8", errors="replace") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            path_s = str(row.get("image_path") or row.get("path") or row.get("image") or "")
            cap = str(row.get("caption") or row.get("text") or "").strip()
            if not path_s:
                n_skip += 1
                continue
            img_path = _resolve_image(path_s, image_root=image_root, manifest_dir=manifest_dir)
            if img_path is None:
                n_bad_path += 1
                continue
            rgb = _load_rgb_uint8(img_path)
            if rgb is None:
                n_bad_path += 1
                continue

            row = dict(row)
            row["aesthetic_proxy"] = round(_aesthetic_proxy(rgb), 6)
            if args.with_phash:
                h = _perceptual_hash_hex(img_path)
                if h:
                    row["dedupe_phash"] = h
            if score_clip_similarity is not None and cap:
                sims = score_clip_similarity([rgb], cap, device=device, model_id=str(args.clip_model))
                row["clip_sim"] = round(float(sims[0]), 6) if sims else 0.0
            elif score_clip_similarity is not None:
                row["clip_sim"] = 0.0

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            if args.max_rows and n_ok >= int(args.max_rows):
                break

    print(
        f"Wrote {n_ok} rows to {out} (skipped_no_path={n_skip}, missing_or_bad_image={n_bad_path})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
