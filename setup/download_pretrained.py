#!/usr/bin/env python3
"""Download pretrained models SDX uses into ``pretrained/``.

Reads ``pretrained_status.json`` (HF repo ids + optional profiles).

    python setup/download_pretrained.py --dest /workspace/pretrained
    python setup/download_pretrained.py --profile train      # ~65 GB, enough to train
    python setup/download_pretrained.py --profile enrich     # VLM + Qwen for captions
    python setup/download_pretrained.py --profile inference  # train + caption + rewards
    python setup/download_pretrained.py --only T5-XXL moondream2

Set ``HF_TOKEN`` (or ``huggingface-cli login``) for gated models.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_JSON = REPO_ROOT / "pretrained_status.json"

_PROFILE_ALIASES = {
    "full": None,
    "all": None,
    "ultimate": None,  # runpod full pipeline — entire pretrained_status.json catalog
    "train": "train",
    "enrich": "enrich",
    "inference": "inference",
    "minimal": "train",
}


def _profile_names(profile: str | None) -> set[str] | None:
    if profile is None:
        return None
    pl = profile.lower()
    if pl not in _PROFILE_ALIASES:
        raise ValueError(f"Unknown profile {profile!r} (use train|enrich|inference|ultimate|full)")
    key = _PROFILE_ALIASES[pl]
    if key is None:
        return None
    names = _load_status().get("profiles", {}).get(key, [])
    if not names:
        raise ValueError(f"Profile {key!r} has no models in pretrained_status.json")
    return {n.lower() for n in names}


def _hf_token() -> str | None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from utils.hf_secrets import apply_hf_token_to_env, get_hf_token, hf_auth_source

        apply_hf_token_to_env()
        return get_hf_token()
    except ImportError:
        pass
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def _retry_wait(exc: BaseException, attempt: int) -> int:
    msg = str(exc)
    m = re.search(r"Retry after (\d+)", msg, re.IGNORECASE)
    if m:
        return int(m.group(1)) + 5
    if "429" in msg or "rate limit" in msg.lower():
        return min(180, 30 * attempt)
    return min(60, 4 * attempt)


def _enable_fast_transfer() -> bool:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        return True
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        return False


def _dir_populated(local_dir: Path, *, min_bytes: int = 1_000_000) -> bool:
    if not local_dir.is_dir():
        return False
    total = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            if total >= min_bytes:
                return True
    return False


def _download_url_files(
    local_dir: Path,
    files: dict[str, str],
    *,
    retries: int,
) -> bool:
    """Download explicit URL -> relative-path mappings (e.g. CodeFormer GitHub releases)."""
    for rel, url in files.items():
        dest = local_dir / rel
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        ok = False
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    with tmp.open("wb") as fh:
                        shutil.copyfileobj(resp, fh)
                tmp.replace(dest)
                ok = True
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                wait = min(60, 4 * attempt)
                print(
                    f"  {rel}: attempt {attempt}/{retries} failed ({type(e).__name__}: {e}); retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
            finally:
                tmp.unlink(missing_ok=True)
        if not ok:
            return False
    return all((local_dir / rel).is_file() for rel in files)


def _download_one(
    repo_id: str,
    local_dir: Path,
    *,
    workers: int,
    retries: int,
    revision: str | None,
    token: str | None,
    allow_patterns: list[str] | None = None,
) -> bool:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), 4 if token else 2))
    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "max_workers": workers,
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(**kwargs)
            return _dir_populated(local_dir)
        except Exception as e:
            wait = _retry_wait(e, attempt)
            print(
                f"  attempt {attempt}/{retries} failed ({type(e).__name__}); "
                f"retrying in {wait}s — set HF_TOKEN in /workspace/secret.txt to avoid 429",
                file=sys.stderr,
            )
            time.sleep(wait)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download SDX pretrained models from pretrained_status.json.")
    p.add_argument("--dest", default=str(REPO_ROOT / "pretrained"), help="Destination base dir.")
    p.add_argument("--only", nargs="*", default=None, help="Download only these model names.")
    p.add_argument(
        "--profile",
        default=os.environ.get("SDX_MODEL_PROFILE", "full"),
        help="train | enrich | inference | ultimate | full (default: full or SDX_MODEL_PROFILE).",
    )
    p.add_argument("--workers", type=int, default=0, help="Parallel file workers per repo (default 4 with HF_TOKEN else 2).")
    p.add_argument("--retries", type=int, default=5, help="Retries per repo on network error.")
    p.add_argument("--force", action="store_true", help="Re-download even when folder looks complete.")
    args = p.parse_args(argv)

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install first: pip install huggingface_hub hf_transfer", file=sys.stderr)
        return 2

    fast = _enable_fast_transfer()
    token = _hf_token()
    print(f"hf_transfer fast download: {'ON' if fast else 'OFF (pip install hf_transfer for a big speedup)'}")
    if token:
        try:
            from utils.hf_secrets import hf_auth_source

            src = hf_auth_source()
            if src == "cli":
                print("HF auth: huggingface-cli login (higher rate limits, gated models OK)")
            elif src == "secret":
                print("HF auth: secret.txt (higher rate limits, gated models OK)")
            else:
                print("HF auth: OK (higher rate limits, gated models OK)")
        except ImportError:
            print("HF auth: OK (higher rate limits, gated models OK)")
    else:
        print(
            "HF auth: NOT SET — you will hit 429 rate limits.\n"
            "  Run: huggingface-cli login\n"
            "  Or add to /workspace/secret.txt:\n"
            "    huggingface\n"
            "    token: hf_your_token_here",
            file=sys.stderr,
        )

    workers = args.workers if args.workers > 0 else (4 if token else 2)

    registry = _load_registry()
    if args.only:
        want = {n.lower() for n in args.only}
        registry = [m for m in registry if m.get("name", "").lower() in want]
    else:
        prof = (args.profile or "full").lower()
        want = _profile_names(prof)
        if want is not None:
            registry = [m for m in registry if m.get("name", "").lower() in want]
            print(f"Profile: {prof} ({len(registry)} models)")
        else:
            print(f"Profile: {prof} (full catalog — {len(registry)} models)")

    if not registry:
        print("No models matched.", file=sys.stderr)
        return 2

    dest_base = Path(args.dest)
    total_gb = sum(float(m.get("size_gb", 0) or 0) for m in registry)
    print(f"Downloading {len(registry)} models (~{total_gb:.1f} GB) -> {dest_base}\n")

    ok, skipped, failed, optional_failed = 0, 0, [], []
    for i, m in enumerate(registry, 1):
        name = m.get("name", "?")
        repo_id = m.get("hf_fallback")
        download_files = m.get("download_files") or {}
        optional = bool(m.get("optional"))
        revision = m.get("hf_revision") or None
        if not repo_id and not download_files:
            print(f"[{i}/{len(registry)}] {name}: no hf_fallback or download_files, skipping", file=sys.stderr)
            continue
        local_dir = dest_base / name
        if not args.force:
            if download_files and all((local_dir / rel).is_file() for rel in download_files):
                print(f"[{i}/{len(registry)}] {name}  <-  github releases  (already present, skip)")
                skipped += 1
                ok += 1
                continue
            if repo_id and _dir_populated(local_dir):
                print(f"[{i}/{len(registry)}] {name}  <-  {repo_id}  (already present, skip)")
                skipped += 1
                ok += 1
                continue
        if download_files:
            print(f"[{i}/{len(registry)}] {name}  <-  github releases  (~{m.get('size_gb', '?')} GB)")
            if _download_url_files(local_dir, download_files, retries=args.retries):
                ok += 1
            else:
                (optional_failed if optional else failed).append(name)
                print(f"  FAILED after {args.retries} retries: {name}", file=sys.stderr)
            continue
        rev_note = f" @{revision}" if revision else ""
        allow = m.get("allow_patterns") or None
        print(f"[{i}/{len(registry)}] {name}  <-  {repo_id}{rev_note}  (~{m.get('size_gb', '?')} GB)")
        if _download_one(
            repo_id,
            local_dir,
            workers=workers,
            retries=args.retries,
            revision=revision,
            token=token,
            allow_patterns=allow,
        ):
            ok += 1
        else:
            (optional_failed if optional else failed).append(name)
            print(f"  FAILED after {args.retries} retries: {name}", file=sys.stderr)
        if not token and i < len(registry):
            time.sleep(10)

    print(f"\nDone: {ok}/{len(registry)} ok ({skipped} skipped as present). Dest: {dest_base}")
    if optional_failed:
        print(f"Optional models failed (non-fatal): {', '.join(optional_failed)}", file=sys.stderr)
    if failed:
        print(f"Required models failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
