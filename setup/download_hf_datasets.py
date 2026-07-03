#!/usr/bin/env python3
"""Export Hugging Face booru-style datasets to SDX layout.

Reads ``setup/hf_dataset_packs.json``. Always exports all four site packs unless
``--only`` is passed.

    python setup/download_hf_datasets.py --dest /workspace/data
    python setup/download_hf_datasets.py --only e621 --max-samples 100000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKS_JSON = REPO_ROOT / "setup" / "hf_dataset_packs.json"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "training" / "hf_export_to_sdx_manifest.py"


def _load_packs() -> list[dict]:
    data = json.loads(PACKS_JSON.read_text(encoding="utf-8"))
    return list(data.get("packs") or [])


def _site_list() -> list[str]:
    if os.environ.get("SDX_HF_SITES", "").strip():
        return [s.strip() for s in os.environ["SDX_HF_SITES"].replace(",", " ").split() if s.strip()]
    return [p["name"] for p in _load_packs()]


def _manifest_ok(site_dir: Path, *, min_rows: int = 1) -> bool:
    m = site_dir / "manifest.jsonl"
    if not m.is_file() or m.stat().st_size < 32:
        return False
    if min_rows <= 1:
        return True
    n = 0
    for line in m.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            n += 1
            if n >= min_rows:
                return True
    return False


def _enable_hf_transfer() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from utils.hf_secrets import apply_hf_token_to_env

        apply_hf_token_to_env()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HF dataset packs -> SDX per-site folders + manifests.")
    p.add_argument("--dest", default=os.environ.get("SDX_DATA", "/workspace/data"), help="Data root.")
    p.add_argument("--only", nargs="*", default=None, help="Subset of pack names (default: SDX_DATA_SITES).")
    p.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("SDX_HF_MAX_SAMPLES", os.environ.get("SDX_MAX_SAMPLES", "0"))),
        help="Per-pack row cap (0 = full stream).",
    )
    p.add_argument("--force", action="store_true", help="Re-export even when manifest exists.")
    p.add_argument("--image-format", default=os.environ.get("SDX_HF_IMAGE_FORMAT", "jpg"), choices=("jpg", "png", "webp"))
    args = p.parse_args(argv)

    _enable_hf_transfer()
    packs = {x["name"]: x for x in _load_packs()}
    want = _site_list()
    if args.only:
        want = [n for n in args.only if n in packs]
    missing = [n for n in want if n not in packs]
    if missing:
        print(f"Unknown pack names: {missing} (known: {sorted(packs)})", file=sys.stderr)
        return 2
    if not want:
        print("No HF dataset packs selected.", file=sys.stderr)
        return 2

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Hugging Face datasets: {len(want)} packs -> {dest}")
    for name in want:
        spec = packs[name]
        label = spec.get("site") or name
        print(f"  {name:12} {label}  <-  {spec['dataset']}")
    print(f"  max_samples per pack: {args.max_samples or 'unlimited'}")
    print(f"  hf_transfer: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '?')}\n")

    ok, skipped = 0, 0
    for name in want:
        spec = packs[name]
        out_dir = dest / name
        if not args.force and _manifest_ok(out_dir):
            print(f"[{name}] already exported — skip ({out_dir / 'manifest.jsonl'})")
            skipped += 1
            ok += 1
            continue

        cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--dataset",
            spec["dataset"],
            "--split",
            str(spec.get("split") or "train"),
            "--image-field",
            str(spec.get("image_field") or "image"),
            "--caption-field",
            str(spec.get("caption_field") or "tag_string"),
            "--out-dir",
            str(out_dir),
            "--streaming",
            "--image-format",
            args.image_format,
        ]
        if spec.get("config"):
            cmd.extend(["--config", str(spec["config"])])
        if spec.get("revision"):
            cmd.extend(["--revision", str(spec["revision"])])
        if spec.get("caption_tag_join"):
            cmd.extend(["--caption-tag-join", str(spec["caption_tag_join"])])
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])

        print(f"[{name}] <- {spec['dataset']}  ->  {out_dir}")
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"  FAILED: {name}", file=sys.stderr)
            continue
        ok += 1

    print(f"\nDone: {ok}/{len(want)} packs ({skipped} skipped as present).")
    return 0 if ok == len(want) else 1


if __name__ == "__main__":
    raise SystemExit(main())
