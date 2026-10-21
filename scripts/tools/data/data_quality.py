#!/usr/bin/env python3
"""
Data quality filter for a JSONL manifest or an image folder (``.txt`` / ``.caption`` sidecars).

Dedup (perceptual hash or file MD5), caption length, bad words, weight / CLIP / aesthetic floors via
:func:`utils.data_quality.filter_jsonl_row`. Writes a filtered JSONL; never modifies the originals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.tools._repo_bootstrap import ensure_repo_paths  # noqa: E402

ensure_repo_paths()

from utils.data_quality import FilterConfig, filter_rows  # noqa: E402


def _folder_rows(folder: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for img_path in folder.rglob(ext):
            cap_path = img_path.with_suffix(".txt")
            if not cap_path.exists():
                cap_path = img_path.with_name(img_path.stem + ".caption")
            cap = (
                cap_path.read_text(encoding="utf-8", errors="ignore").strip().split("\n")[0]
                if cap_path.exists()
                else ""
            )
            yield {"image_path": str(img_path), "caption": cap}


def _native_checks(args: argparse.Namespace, inp: Path) -> int | None:
    """Rust ``sdx-jsonl-tools`` stats / validate before filtering; an exit code stops the run."""
    from utils.native import run_rust_jsonl_stats, run_rust_jsonl_validate, rust_jsonl_tools_exe

    if rust_jsonl_tools_exe() is None:
        print("native: Rust sdx-jsonl-tools not built; skipping. See native/README.md", file=sys.stderr)
        return 1 if args.native_stats else None
    if args.native_preflight or args.native_stats:
        rv = run_rust_jsonl_stats(inp)
        sys.stderr.write("[native] sdx-jsonl-tools stats:\n" + (rv.stdout or "") + (rv.stderr or ""))
        if args.native_stats or rv.returncode != 0:
            return rv.returncode
    if args.native_validate:
        rv = run_rust_jsonl_validate(inp, min_caption_len=args.min_caption_len, max_caption_len=args.max_caption_len)
        sys.stderr.write(rv.stderr or "")
        sys.stdout.write(rv.stdout or "")
        if rv.returncode != 0:
            print("native-validate: Rust validate failed; fix manifest or adjust caption bounds", file=sys.stderr)
            return rv.returncode or 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter/dedup JSONL or scan folder for data quality.")
    parser.add_argument("input", type=str, help="Path to manifest JSONL or folder of images")
    parser.add_argument("--out", type=str, default="", help="Output filtered JSONL (default: <input>_quality.jsonl)")
    parser.add_argument(
        "--dedup", type=str, default="", choices=["", "phash", "md5"], help="Dedup by perceptual hash or file MD5"
    )
    parser.add_argument("--min-caption-len", type=int, default=0, help="Drop rows with caption length < N")
    parser.add_argument("--max-caption-len", type=int, default=0, help="Drop rows with caption length > N (0=off)")
    parser.add_argument("--bad-words", type=str, default="", help="Comma-sep words; drop if caption contains any")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Drop rows with weight/aesthetic_score < N")
    parser.add_argument(
        "--min-clip-sim",
        type=float,
        default=0.0,
        help="If >0: drop rows whose clip_sim is below this (rows without clip_sim pass). Run manifest_enrich first.",
    )
    parser.add_argument(
        "--min-aesthetic-proxy",
        type=float,
        default=0.0,
        help="If >0: drop rows whose aesthetic_proxy is below this (rows without it pass). Run manifest_enrich first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not write output")
    parser.add_argument(
        "--native-preflight",
        action="store_true",
        help="If Rust sdx-jsonl-tools is built, print fast `stats` on stderr before filtering.",
    )
    parser.add_argument(
        "--native-validate",
        action="store_true",
        help="If Rust is built, run strict `validate` with --min/--max-caption-len and exit non-zero on any bad row.",
    )
    parser.add_argument(
        "--native-stats",
        action="store_true",
        help="Print Rust JSONL `stats` for a .jsonl input and exit (ignores folder mode and other filters).",
    )
    parser.add_argument(
        "--no-native-md5",
        action="store_true",
        help="For --dedup md5: use Python hashlib only (skip Rust file-md5 subprocess).",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1
    is_jsonl = inp.suffix.lower() == ".jsonl"
    if is_jsonl and (args.native_preflight or args.native_validate or args.native_stats):
        rc = _native_checks(args, inp)
        if rc is not None:
            return rc

    from data.manifest_utils import iter_manifest_rows

    cfg = FilterConfig(
        dedup=args.dedup,
        min_caption_len=args.min_caption_len,
        max_caption_len=args.max_caption_len,
        bad_words=tuple(w.strip().lower() for w in args.bad_words.split(",") if w.strip()),
        min_weight=args.min_weight,
        min_clip_sim=args.min_clip_sim,
        min_aesthetic_proxy=args.min_aesthetic_proxy,
        native_md5=not args.no_native_md5,
    )
    source = iter_manifest_rows(inp) if is_jsonl else _folder_rows(inp)
    rows, stats = filter_rows(source, cfg, base_dir=inp.parent if is_jsonl else inp)

    print(
        f"Kept: {len(rows)}, dropped: dup={stats.dropped_dup} caption={stats.dropped_caption} "
        f"bad_words={stats.dropped_bad_words} weight={stats.dropped_weight} clip_sim={stats.dropped_clip} "
        f"aesthetic_proxy={stats.dropped_aesthetic}",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return 0
    out_path = Path(args.out) if args.out else inp.parent / (inp.stem + "_quality.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
