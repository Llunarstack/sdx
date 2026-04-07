#!/usr/bin/env python3
"""
Data quality pipeline for JSONL or image folders (IMPROVEMENTS 1.6).
- Dedup: by perceptual hash (imagehash) or file MD5.
- Filter: min/max caption length, bad-words list, optional weight/aesthetic column.
- Output: filtered JSONL. Does not modify originals.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_NATIVE_PY = ROOT / "native" / "python"
if str(_NATIVE_PY) not in sys.path:
    sys.path.insert(0, str(_NATIVE_PY))


def _perceptual_hash(path: Path, size: int = 8) -> str:
    """Perceptual hash of image (requires imagehash + PIL). Returns hex string or empty on error."""
    try:
        import imagehash
        from PIL import Image

        phash = imagehash.phash(Image.open(path), hash_size=size)
        return str(phash)
    except ImportError:
        return ""
    except Exception:
        return ""


def _file_hash(path: Path, *, prefer_native_md5: bool = True) -> str:
    """
    MD5 of file bytes (streaming). When Rust ``sdx-jsonl-tools`` is built, uses ``file-md5``
    (1 MiB chunks, no giant Python allocations). Otherwise ``hashlib`` in 1 MiB chunks.
    """
    try:
        from sdx_native.native_tools import file_md5_hex

        return file_md5_hex(path, prefer_native_md5=prefer_native_md5)
    except Exception:
        pass
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Filter/dedup JSONL or scan folder for data quality.")
    parser.add_argument("input", type=str, help="Path to manifest JSONL or folder of images")
    parser.add_argument(
        "--out", type=str, default="", help="Output filtered JSONL (default: print to stdout or input_quality.jsonl)"
    )
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
        help="If >0: drop rows that have clip_sim below this (ignores rows without clip_sim). Use manifest_enrich first.",
    )
    parser.add_argument(
        "--min-aesthetic-proxy",
        type=float,
        default=0.0,
        help="If >0: drop rows with aesthetic_proxy below this (ignores rows without key). Use manifest_enrich first.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not write output")
    parser.add_argument(
        "--native-preflight",
        action="store_true",
        help="If Rust sdx-jsonl-tools is built, print fast `stats` on stderr before Python filtering (JSON/caption distribution).",
    )
    parser.add_argument(
        "--native-validate",
        action="store_true",
        help="If Rust is built, run strict `validate` with --min/--max-caption-len and exit non-zero on any bad row (before filtering).",
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

    bad_set = set(x.strip().lower() for x in args.bad_words.split(",") if x.strip()) if args.bad_words else set()
    seen_hashes = set()
    rows = []
    dropped_dup = dropped_caption = dropped_bad = dropped_weight = dropped_clip = dropped_aes = 0
    prefer_native_md5 = not args.no_native_md5

    if inp.suffix.lower() == ".jsonl":
        if args.native_preflight or args.native_validate:
            try:
                from sdx_native.native_tools import run_rust_jsonl_stats, run_rust_jsonl_validate, rust_jsonl_tools_exe

                exe = rust_jsonl_tools_exe()
                if exe:
                    if args.native_preflight:
                        rv = run_rust_jsonl_stats(inp)
                        sys.stderr.write("[native] sdx-jsonl-tools stats:\n" + (rv.stdout or ""))
                        if rv.stderr:
                            sys.stderr.write(rv.stderr)
                        if rv.returncode != 0:
                            return rv.returncode
                    if args.native_validate:
                        rv = run_rust_jsonl_validate(
                            inp,
                            min_caption_len=args.min_caption_len,
                            max_caption_len=args.max_caption_len,
                        )
                        sys.stderr.write(rv.stderr or "")
                        sys.stdout.write(rv.stdout or "")
                        if rv.returncode != 0:
                            print(
                                "native-validate: Rust validate failed — fix manifest or adjust caption bounds",
                                file=sys.stderr,
                            )
                            return rv.returncode or 1
                else:
                    print(
                        "native: Rust sdx-jsonl-tools not built; skipping. See native/README.md",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"native tools: {e}", file=sys.stderr)
                return 1
        with open(inp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                path = d.get("image_path") or d.get("path") or d.get("image") or ""
                cap = (d.get("caption") or d.get("text") or "").strip()
                if not path or not cap:
                    continue
                # Dedup
                if args.dedup:
                    img_path = Path(path)
                    if not img_path.is_absolute():
                        img_path = (inp.parent / img_path).resolve()
                    h = (
                        _perceptual_hash(img_path)
                        if args.dedup == "phash"
                        else _file_hash(img_path, prefer_native_md5=prefer_native_md5)
                    )
                    if not h and args.dedup == "phash":
                        h = _file_hash(img_path, prefer_native_md5=prefer_native_md5)
                    if h and h in seen_hashes:
                        dropped_dup += 1
                        continue
                    if h:
                        seen_hashes.add(h)
                # Caption length
                if args.min_caption_len and len(cap) < args.min_caption_len:
                    dropped_caption += 1
                    continue
                if args.max_caption_len and len(cap) > args.max_caption_len:
                    dropped_caption += 1
                    continue
                # Bad words
                if bad_set and any(w in cap.lower() for w in bad_set):
                    dropped_bad += 1
                    continue
                # Weight
                w = float(d.get("weight", d.get("aesthetic_score", 1.0)))
                if args.min_weight > 0 and w < args.min_weight:
                    dropped_weight += 1
                    continue
                if args.min_clip_sim > 0.0 and "clip_sim" in d:
                    try:
                        if float(d["clip_sim"]) < float(args.min_clip_sim):
                            dropped_clip += 1
                            continue
                    except (TypeError, ValueError):
                        pass
                if args.min_aesthetic_proxy > 0.0 and "aesthetic_proxy" in d:
                    try:
                        if float(d["aesthetic_proxy"]) < float(args.min_aesthetic_proxy):
                            dropped_aes += 1
                            continue
                    except (TypeError, ValueError):
                        pass
                rows.append(d)
    else:
        # Folder: list images and optional .txt captions (no dedup by caption across files)
        caption_ext = ".txt"
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_path in inp.rglob(ext):
                cap_path = img_path.with_suffix(caption_ext)
                if not cap_path.exists():
                    cap_path = img_path.with_name(img_path.stem + ".caption")
                cap = (
                    cap_path.read_text(encoding="utf-8", errors="ignore").strip().split("\n")[0]
                    if cap_path.exists()
                    else ""
                )
                path_str = str(img_path)
                if args.dedup:
                    h = (
                        _perceptual_hash(img_path)
                        if args.dedup == "phash"
                        else _file_hash(img_path, prefer_native_md5=prefer_native_md5)
                    )
                    if not h and args.dedup == "phash":
                        h = _file_hash(img_path, prefer_native_md5=prefer_native_md5)
                    if h and h in seen_hashes:
                        dropped_dup += 1
                        continue
                    if h:
                        seen_hashes.add(h)
                if args.min_caption_len and len(cap) < args.min_caption_len:
                    dropped_caption += 1
                    continue
                if args.max_caption_len and len(cap) > args.max_caption_len:
                    dropped_caption += 1
                    continue
                if bad_set and cap and any(w in cap.lower() for w in bad_set):
                    dropped_bad += 1
                    continue
                rows.append({"image_path": path_str, "caption": cap})

    print(
        f"Kept: {len(rows)}, dropped: dup={dropped_dup} caption={dropped_caption} bad_words={dropped_bad} "
        f"weight={dropped_weight} clip_sim={dropped_clip} aesthetic_proxy={dropped_aes}",
        file=sys.stderr,
    )
    if args.dry_run:
        return 0
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return 0
    out_path = Path(args.out) if args.out else inp.parent / (inp.stem + "_quality.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
