#!/usr/bin/env python3
"""Generate control maps (canny, softedge, depth) for ControlNet training.

Reads a training manifest, writes ``controls/<type>/<md5>.png`` beside each image,
and emits a new manifest where every row has ``control_image`` + ``control_type``.

    python setup/preprocess_control_maps.py --manifest /workspace/data/combined/manifest.jsonl \\
        --data-root /workspace/data --control-type canny --out /workspace/data/control/manifest.jsonl

Training::

    python train.py --manifest-jsonl /workspace/data/control/manifest.jsonl \\
        --control-cond-dim 1 --control-num-types 9 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_image(data_root: Path, row: dict) -> Path | None:
    rel = row.get("image_path") or row.get("path") or ""
    if not rel:
        return None
    p = Path(rel)
    if p.is_file():
        return p
    full = data_root / p
    if full.is_file():
        return full
    return None


def _extract_one(
    data_root: Path,
    row: dict,
    *,
    control_type: str,
    controls_dir: Path,
) -> dict | None:
    img = _resolve_image(data_root, row)
    if img is None:
        return None
    md5 = row.get("md5") or img.stem.split("_f")[0]
    dest = controls_dir / control_type / f"{md5}.png"
    if dest.is_file():
        pass
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from utils.modeling.hf_control import extract_control_maps_batch

            maps = extract_control_maps_batch(str(img), dest.parent, types=(control_type,))
            if control_type not in maps:
                from utils.modeling.hf_control import extract_pil_proxy

                out = extract_pil_proxy(str(img), str(dest), control_type)
                if not out:
                    return None
            elif maps.get(control_type) != str(dest):
                # batch writes stem_type.png — rename to md5 if needed
                src = Path(maps[control_type])
                if src.is_file() and src != dest:
                    src.replace(dest)
        except Exception:
            try:
                import cv2

                rgb = Image.open(img).convert("RGB")
                arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(arr, 80, 160)
                cv2.imwrite(str(dest), edges)
            except Exception:
                return None

    rel_ctrl = dest.relative_to(data_root).as_posix()
    out_row = dict(row)
    out_row["control_image"] = rel_ctrl
    out_row["control_type"] = control_type
    return out_row


def preprocess(
    manifest: Path,
    data_root: Path,
    out_manifest: Path,
    *,
    control_type: str = "canny",
    workers: int = 8,
    max_rows: int = 0,
) -> int:
    controls_dir = data_root / "controls"
    written = 0
    rows_in = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if max_rows > 0:
        rows_in = rows_in[:max_rows]

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    _extract_one,
                    data_root,
                    row,
                    control_type=control_type,
                    controls_dir=controls_dir,
                ): row
                for row in rows_in
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build control-paired manifest for ControlNet training.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--control-type", default="canny", choices=["canny", "softedge", "hed", "depth", "normals"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all).")
    args = p.parse_args(argv)

    n = preprocess(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.out),
        control_type=args.control_type,
        workers=args.workers,
        max_rows=args.max_rows,
    )
    print(f"Wrote {n:,} control-paired rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
