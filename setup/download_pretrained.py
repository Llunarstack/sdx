#!/usr/bin/env python3
"""Download every pretrained model SDX uses, into ``pretrained/``.

Reads the model registry from ``pretrained_status.json`` (name + HF fallback id)
and pulls each via ``huggingface_hub.snapshot_download``. Fast and robust:

  * Enables ``hf_transfer`` (Rust multi-connection downloader) if installed.
  * Resumable — snapshot_download skips files already present.
  * Retries each repo with backoff so a transient network blip doesn't abort the run.

    python setup/download_pretrained.py                     # all models
    python setup/download_pretrained.py --only T5-XXL CLIP-ViT-L-14
    python setup/download_pretrained.py --dest /workspace/pretrained --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = REPO_ROOT / "pretrained_status.json"


def _load_registry() -> list[dict]:
    data = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    return data.get("models", [])


def _enable_fast_transfer() -> bool:
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        return False


def _download_one(repo_id: str, local_dir: Path, workers: int, retries: int) -> bool:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            # snapshot_download resumes by default and skips already-present files.
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir), max_workers=workers)
            return True
        except Exception as e:  # network/HTTP/filesystem — retry with backoff
            wait = min(60, 4 * attempt)
            print(f"  attempt {attempt}/{retries} failed ({type(e).__name__}: {e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download all SDX pretrained models from pretrained_status.json.")
    p.add_argument("--dest", default=str(REPO_ROOT / "pretrained"), help="Destination base dir (default: pretrained/).")
    p.add_argument("--only", nargs="*", default=None, help="Download only these model names (default: all).")
    p.add_argument("--workers", type=int, default=8, help="Parallel file workers per repo.")
    p.add_argument("--retries", type=int, default=5, help="Retries per repo on network error.")
    args = p.parse_args(argv)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install first: pip install huggingface_hub hf_transfer", file=sys.stderr)
        return 2

    fast = _enable_fast_transfer()
    print(f"hf_transfer fast download: {'ON' if fast else 'OFF (pip install hf_transfer for a big speedup)'}")

    registry = _load_registry()
    if args.only:
        want = {n.lower() for n in args.only}
        registry = [m for m in registry if m.get("name", "").lower() in want]
        if not registry:
            print(f"No registry entries matched {args.only}", file=sys.stderr)
            return 2

    dest_base = Path(args.dest)
    total_gb = sum(float(m.get("size_gb", 0) or 0) for m in registry)
    print(f"Downloading {len(registry)} models (~{total_gb:.1f} GB) -> {dest_base}\n")

    ok, failed = 0, []
    for i, m in enumerate(registry, 1):
        name = m.get("name", "?")
        repo_id = m.get("hf_fallback")
        if not repo_id:
            print(f"[{i}/{len(registry)}] {name}: no hf_fallback id, skipping", file=sys.stderr)
            continue
        local_dir = dest_base / name
        print(f"[{i}/{len(registry)}] {name}  <-  {repo_id}  (~{m.get('size_gb', '?')} GB)")
        if _download_one(repo_id, local_dir, args.workers, args.retries):
            ok += 1
        else:
            failed.append(name)
            print(f"  FAILED after {args.retries} retries: {name}", file=sys.stderr)

    print(f"\nDone: {ok}/{len(registry)} models. Dest: {dest_base}")
    if failed:
        print(f"Failed: {', '.join(failed)} (rerun to resume — completed files are skipped).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
