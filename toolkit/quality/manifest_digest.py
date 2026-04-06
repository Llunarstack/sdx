#!/usr/bin/env python3
"""
Quick JSONL manifest stats without loading images (stdlib + optional Rust).

Usage:
  python -m toolkit.quality.manifest_digest path/to/manifest.jsonl
  python -m toolkit.quality.manifest_digest path/to/manifest.jsonl --rust-stats
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def digest_jsonl(path: Path, *, max_key_samples: int = 12) -> Dict[str, Any]:
    n = 0
    key_counts: Counter[str] = Counter()
    caption_keys = ("caption", "text", "prompt")
    has_image = 0
    has_caption = 0
    sample_paths: List[str] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            for k in row.keys():
                key_counts[k] += 1
            ip = row.get("image_path") or row.get("path") or row.get("image")
            if isinstance(ip, str) and ip.strip():
                has_image += 1
                if len(sample_paths) < 3:
                    sample_paths.append(ip.strip()[:120])
            cap = None
            for ck in caption_keys:
                if ck in row and isinstance(row[ck], str):
                    cap = row[ck]
                    break
            if cap and cap.strip():
                has_caption += 1

    top_keys = [k for k, _ in key_counts.most_common(max_key_samples)]
    return {
        "path": str(path.resolve()),
        "lines_read": n,
        "rows_with_image_field": has_image,
        "rows_with_caption_like": has_caption,
        "distinct_keys_top": top_keys,
        "sample_image_paths": sample_paths,
    }


def try_rust_stats(manifest: Path) -> str | None:
    """Run ``sdx-jsonl-tools stats`` when the release binary exists (via ``native_tools``)."""
    try:
        from utils.native import run_rust_jsonl_stats, rust_jsonl_tools_exe
    except ImportError:
        return None
    if rust_jsonl_tools_exe() is None:
        return None
    try:
        r = run_rust_jsonl_stats(manifest, timeout=600)
        if r.returncode != 0:
            return r.stderr or r.stdout or "rust stats failed"
        return (r.stdout or "").strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="JSONL manifest digest (quick QoL)")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--rust-stats", action="store_true", help="Append sdx-jsonl-tools stats if built")
    p.add_argument("--json-out", action="store_true", help="Print digest as JSON only")
    args = p.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    d = digest_jsonl(path)
    if args.rust_stats:
        d["rust_stats_stdout"] = try_rust_stats(path)

    if args.json_out:
        print(json.dumps(d, indent=2))
        return 0

    print(f"=== manifest_digest: {d['path']} ===")
    print(f"lines_read: {d['lines_read']}")
    print(f"rows_with_image_field: {d['rows_with_image_field']}")
    print(f"rows_with_caption_like: {d['rows_with_caption_like']}")
    print(f"top_keys: {d['distinct_keys_top']}")
    if d["sample_image_paths"]:
        print("sample paths:", d["sample_image_paths"])
    if args.rust_stats:
        print("--- rust stats ---")
        print(d.get("rust_stats_stdout") or "(sdx-jsonl-tools not built or failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
