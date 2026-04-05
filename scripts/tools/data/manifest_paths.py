#!/usr/bin/env python3
"""
Extract image paths from a JSONL manifest (for shell pipelines / disk audits).

Uses Rust **sdx-jsonl-tools image-paths** if built; otherwise pure Python.

Examples::

    python -m scripts.tools manifest_paths data/train.jsonl
    python -m scripts.tools manifest_paths data/train.jsonl --sort > paths.txt
    python -m scripts.tools manifest_paths data/train.jsonl --dup
    # Pipe path list to Zig pathstat (if built):
    #   python -m scripts.tools manifest_paths m.jsonl | native/zig/sdx-pathstat/zig-out/bin/sdx-pathstat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.native.native_tools import (  # noqa: E402
    run_rust_dup_image_paths,
    run_rust_image_paths,
    rust_jsonl_tools_exe,
)


def _python_image_paths(manifest: Path, *, all_rows: bool, sort_paths: bool) -> str:
    seen: set[str] = set()
    out: list[str] = []
    with manifest.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            p = ""
            for k in ("image_path", "path", "image"):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    p = v.strip()
                    break
            if not p:
                continue
            if all_rows:
                out.append(p)
            elif p not in seen:
                seen.add(p)
                out.append(p)
    if sort_paths:
        out.sort()
    return "\n".join(out) + ("\n" if out else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract image paths from SDX JSONL manifests.")
    ap.add_argument("manifest", type=str, help="Path to .jsonl")
    ap.add_argument("--all-rows", action="store_true", help="Emit path per row (allow duplicates)")
    ap.add_argument("--sort", action="store_true", help="Sort paths lexicographically")
    ap.add_argument(
        "--dup",
        action="store_true",
        help="Print duplicate path report (Rust dup-image-paths if built)",
    )
    args = ap.parse_args()

    mp = Path(args.manifest)
    if not mp.is_file():
        print(f"Not found: {mp}", file=sys.stderr)
        return 2

    exe = rust_jsonl_tools_exe()
    if args.dup:
        if exe:
            r = run_rust_dup_image_paths(mp, top=50)
            sys.stdout.write(r.stdout)
            if r.stderr:
                sys.stderr.write(r.stderr)
            return r.returncode
        print("dup: build Rust sdx-jsonl-tools first.", file=sys.stderr)
        return 1

    if exe:
        r = run_rust_image_paths(mp, all_rows=args.all_rows, sort=args.sort)
        sys.stdout.write(r.stdout or "")
        if r.stderr:
            sys.stderr.write(r.stderr)
        return r.returncode

    sys.stdout.write(_python_image_paths(mp, all_rows=args.all_rows, sort_paths=args.sort))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
