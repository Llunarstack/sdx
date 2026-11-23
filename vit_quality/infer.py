#!/usr/bin/env python3
"""Score a JSONL manifest with a ViT quality/adherence checkpoint.

``--out`` writes rows with ``vit_quality_prob`` / ``vit_adherence_score`` (``--rank`` sorts them by
``vit_final_score``); ``--out-npz`` exports the fused embeddings for retrieval/reranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import json_dumps, json_loads


def _manifest_rows(manifest: Path):
    """Yield ``(row, image_path, caption)`` for readable rows whose image exists."""
    with manifest.open("r", encoding="utf-8") as rf:
        for line in rf:
            t = line.strip()
            if not t:
                continue
            try:
                row = json_loads(t)
            except Exception:
                continue
            image_path = row.get("image_path") or row.get("path") or row.get("image")
            if not isinstance(image_path, str) or not image_path:
                continue
            p = Path(image_path)
            if not p.is_absolute():
                p = (manifest.parent / p).resolve()
            if p.exists():
                yield row, p, str(row.get("caption") or row.get("text") or "")


def main() -> int:
    from utils.modeling.architecture import normalize_num_ar_blocks, parse_num_ar_blocks_from_row

    from vit_quality.model import ViTScorer

    p = argparse.ArgumentParser(description="Run ViT quality/adherence scoring on a JSONL manifest")
    p.add_argument("--ckpt", required=True, help="Path to ViT checkpoint (best.pt/last.pt)")
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--out", default="", help="Output JSONL with appended quality/adherence scores")
    p.add_argument("--out-npz", default="", help="Export fused embeddings + image paths to this .npz")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tta", action="store_true", help="Average predictions with the horizontal flip")
    p.add_argument("--use-ema", action="store_true", help="If checkpoint contains ema_state_dict, use it")
    p.add_argument(
        "--default-num-ar-blocks",
        type=int,
        default=None,
        help="When AR conditioning is on and a row has no AR field, use this (0/2/4) instead of 'unknown'",
    )
    p.add_argument("--rank", action="store_true", help="Sort --out rows by vit_final_score (descending)")
    p.add_argument("--quality-weight", type=float, default=0.6, help="vit_final_score weight on quality")
    p.add_argument("--adherence-weight", type=float, default=0.4, help="vit_final_score weight on adherence")
    p.add_argument("--top-k", type=int, default=0, help="With --rank: keep only the best K rows (0 = all)")
    args = p.parse_args()
    if not args.out and not args.out_npz:
        p.error("pass --out and/or --out-npz")

    scorer = ViTScorer(args.ckpt, args.device, use_ema=args.use_ema)
    default_ar = normalize_num_ar_blocks(args.default_num_ar_blocks)
    manifest = Path(args.manifest_jsonl)
    rows: list[dict] = []
    vecs: list[np.ndarray] = []
    paths: list[str] = []
    for row, img_path, caption in _manifest_rows(manifest):
        try:
            img = Image.open(img_path)
            x = scorer.image_tensor(img).unsqueeze(0)
        except Exception:
            continue
        ar_b = parse_num_ar_blocks_from_row(row)
        out = scorer(x, caption, ar_b if ar_b != -1 else default_ar, tta=args.tta)
        q = float(torch.sigmoid(out["quality_logit"]).item())
        a = float(out["adherence_score"].item())
        row["vit_quality_prob"] = q
        row["vit_adherence_score"] = a
        row["vit_final_score"] = args.quality_weight * q + args.adherence_weight * a
        rows.append(row)
        if args.out_npz:
            vecs.append(out["embedding"].squeeze(0).float().cpu().numpy())
            paths.append(str(img_path))

    if args.out:
        if args.rank:
            rows.sort(key=lambda r: r["vit_final_score"], reverse=True)
            if args.top_k > 0:
                rows = rows[: args.top_k]
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as wf:
            for row in rows:
                wf.write(json_dumps(row, ensure_ascii=False) + "\n")
        print(f"[ViT] wrote {len(rows)} rows -> {outp}")
    if args.out_npz:
        if not vecs:
            print("No valid rows for embedding export.", file=sys.stderr)
            return 2
        arr = np.stack(vecs, axis=0)
        outp = Path(args.out_npz)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outp, embeddings=arr, image_paths=np.array(paths, dtype=object))
        print(f"[ViT] exported {arr.shape[0]} embeddings ({arr.shape[1]} dim) -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
