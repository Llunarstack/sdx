#!/usr/bin/env python3
"""Export Hugging Face booru-style datasets to SDX layout (turbo: parallel packs + threaded writes)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    os.environ.setdefault("HF_HUB_DOWNLOAD_NUM_THREADS", os.environ.get("HF_HUB_DOWNLOAD_NUM_THREADS", "32"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from utils.hf_secrets import apply_hf_token_to_env

        apply_hf_token_to_env()
    except Exception:
        pass


def _export_cmd(spec: dict, *, dest: Path, max_samples: int, image_format: str, turbo: bool) -> list[str]:
    out_dir = dest / spec["name"]
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
        image_format,
    ]
    if spec.get("config"):
        cmd.extend(["--config", str(spec["config"])])
    if spec.get("revision"):
        cmd.extend(["--revision", str(spec["revision"])])
    if spec.get("caption_tag_join"):
        cmd.extend(["--caption-tag-join", str(spec["caption_tag_join"])])
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])
    if turbo or os.environ.get("SDX_HF_TURBO", "").strip() in ("1", "true", "yes"):
        cmd.append("--turbo")
    return cmd


def _run_pack(
    name: str, spec: dict, *, dest: Path, max_samples: int, image_format: str, force: bool, turbo: bool
) -> tuple[str, int]:
    out_dir = dest / name
    if not force and _manifest_ok(out_dir):
        print(f"[{name}] already exported — skip ({out_dir / 'manifest.jsonl'})")
        return name, 0
    cmd = _export_cmd(spec, dest=dest, max_samples=max_samples, image_format=image_format, turbo=turbo)
    print(f"[{name}] <- {spec['dataset']}  ->  {out_dir}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return name, r.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HF dataset packs -> SDX per-site folders + manifests.")
    p.add_argument("--dest", default=os.environ.get("SDX_DATA", "/workspace/data"), help="Data root.")
    p.add_argument("--only", nargs="*", default=None, help="Subset of pack names.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=int(os.environ.get("SDX_HF_MAX_SAMPLES", os.environ.get("SDX_MAX_SAMPLES", "0"))),
        help="Per-pack row cap (0 = full stream).",
    )
    p.add_argument("--force", action="store_true", help="Re-export even when manifest exists.")
    p.add_argument(
        "--image-format", default=os.environ.get("SDX_HF_IMAGE_FORMAT", "jpg"), choices=("jpg", "png", "webp")
    )
    p.add_argument("--parallel", type=int, default=0, help="Export N packs at once (0=SDX_HF_PARALLEL_PACKS).")
    args = p.parse_args(argv)

    _enable_hf_transfer()
    turbo = os.environ.get("SDX_HF_TURBO", "1").strip() in ("1", "true", "yes")
    parallel = args.parallel or int(os.environ.get("SDX_HF_PARALLEL_PACKS", "1") or 1)
    force = args.force or os.environ.get("SDX_HF_FORCE", "").strip() in ("1", "true", "yes")

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
    workers = os.environ.get("SDX_HF_EXPORT_WORKERS", "?")
    print(f"Hugging Face datasets: {len(want)} packs -> {dest}")
    for name in want:
        spec = packs[name]
        print(f"  {name:12} {spec.get('site') or name}  <-  {spec['dataset']}")
    print(f"  max_samples per pack: {args.max_samples or 'unlimited'}")
    print(f"  turbo: {turbo}  parallel_packs: {parallel}  export_workers: {workers}")
    print(f"  hf_transfer: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '?')}\n")

    todo = []
    skipped = 0
    for name in want:
        out_dir = dest / name
        if not force and _manifest_ok(out_dir):
            print(f"[{name}] already exported — skip ({out_dir / 'manifest.jsonl'})")
            skipped += 1
            continue
        todo.append((name, packs[name]))

    if not todo:
        print(f"\nDone: {len(want)}/{len(want)} packs ({skipped} skipped as present).")
        return 0

    ok = skipped
    if parallel > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=min(parallel, len(todo))) as pool:
            futs = {
                pool.submit(
                    _run_pack,
                    name,
                    spec,
                    dest=dest,
                    max_samples=args.max_samples,
                    image_format=args.image_format,
                    force=True,
                    turbo=turbo,
                ): name
                for name, spec in todo
            }
            for fut in as_completed(futs):
                name, rc = fut.result()
                if rc == 0:
                    ok += 1
                else:
                    print(f"  FAILED: {name}", file=sys.stderr)
    else:
        for name, spec in todo:
            _, rc = _run_pack(
                name,
                spec,
                dest=dest,
                max_samples=args.max_samples,
                image_format=args.image_format,
                force=True,
                turbo=turbo,
            )
            if rc == 0:
                ok += 1
            else:
                print(f"  FAILED: {name}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(want)} packs ({skipped} skipped as present).")
    return 0 if ok == len(want) else 1


if __name__ == "__main__":
    raise SystemExit(main())
