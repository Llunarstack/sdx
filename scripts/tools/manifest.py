#!/usr/bin/env python3
"""JSONL training-manifest tools: filter, enrich, gate, merge, inspect, export from Hugging Face.

    python -m scripts.tools manifest quality data/train.jsonl --dedup phash --min-caption-len 10
    python -m scripts.tools manifest enrich data/train.jsonl --with-phash
    python -m scripts.tools manifest gate data/train.jsonl --report-dups --fail-on-dup-groups 5
    python -m scripts.tools manifest image-qc data/train.jsonl --min-sharpness 0.02
    python -m scripts.tools manifest paths data/train.jsonl --sort > paths.txt
    python -m scripts.tools manifest merge -o merged.jsonl a.jsonl b.jsonl
    python -m scripts.tools manifest ar-tag --dit-ckpt best.pt --manifest-jsonl in.jsonl --out out.jsonl
    python -m scripts.tools manifest corpus build --jsonl data/danbooru.jsonl --out data/corpus_index.jsonl
    python -m scripts.tools manifest corpus query --index data/corpus_index.jsonl --prompt "1girl, catgirl"
    python -m scripts.tools manifest smoke-dataset --out data/smoke_tiny
    python -m scripts.tools manifest hf-export --dataset ORG/name --out-dir data/hf_danbooru --max-samples 5000
    python -m scripts.tools manifest hf-train --dataset ORG/name --max-samples 500 -- --max-steps 50

Old command names still work: ``data_quality``, ``manifest_enrich``, ``manifest_gate``, ``image_quality_qc``,
``manifest_paths``, ``jsonl_merge``, ``ar_tag_manifest``, ``corpus_retrieve``, ``make_smoke_dataset``,
``hf_export_to_sdx_manifest``, ``hf_download_and_train``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Queue
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.tools._repo_bootstrap import ensure_repo_paths  # noqa: E402

ensure_repo_paths()

from data.manifest_utils import first_str, iter_manifest_rows, resolve_image_path  # noqa: E402


def _prog(cmd: str) -> str:
    return f"python -m scripts.tools manifest {cmd}"


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _dict_rows(path: Path) -> Iterator[dict]:
    return (r for r in iter_manifest_rows(path) if isinstance(r, dict))


def _row_image(row: dict) -> str:
    return first_str(row, ("image_path", "path", "image"))


# --- quality ---------------------------------------------------------------------------------


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


def _quality_main(argv: list[str]) -> int:
    from utils.quality.critics import FilterConfig, filter_rows

    parser = argparse.ArgumentParser(
        prog=_prog("quality"), description="Filter/dedup JSONL or scan folder for data quality."
    )
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
        help="If >0: drop rows whose clip_sim is below this (rows without clip_sim pass). Run `manifest enrich` first.",
    )
    parser.add_argument(
        "--min-aesthetic-proxy",
        type=float,
        default=0.0,
        help="If >0: drop rows whose aesthetic_proxy is below this (rows without it pass). Run `manifest enrich` first.",
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
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1
    is_jsonl = inp.suffix.lower() == ".jsonl"
    if is_jsonl and (args.native_preflight or args.native_validate or args.native_stats):
        rc = _native_checks(args, inp)
        if rc is not None:
            return rc

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
    _write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


# --- enrich ----------------------------------------------------------------------------------


def _load_rgb_uint8(path: Path):
    try:
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _enrich_main(argv: list[str]) -> int:
    from utils.quality.critics import perceptual_hash
    from utils.quality.test_time_pick import score_aesthetic_proxy

    ap = argparse.ArgumentParser(
        prog=_prog("enrich"),
        description="Add curation signals to a manifest JSONL: clip_sim (CLIP image-text similarity), "
        "aesthetic_proxy (heuristic [0,1]: exposure, tiling, dynamic range), optional dedupe_phash.",
    )
    ap.add_argument("input", type=str, help="Input JSONL manifest")
    ap.add_argument("--out", type=str, default="", help="Output JSONL (default: input stem + _enriched.jsonl)")
    ap.add_argument("--image-root", type=str, default="", help="Base dir for relative image paths")
    ap.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP id (same family as sample.py / test_time_pick)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-rows", type=int, default=0, help="Stop after N valid rows (0 = all)")
    ap.add_argument("--skip-clip", action="store_true", help="Only add aesthetic_proxy / phash (no transformers)")
    ap.add_argument("--with-phash", action="store_true", help="Add dedupe_phash column (needs imagehash + PIL)")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else inp.parent / f"{inp.stem}_enriched.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root).resolve() if args.image_root else None

    device = args.device
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
    score_clip_similarity = None
    if not args.skip_clip:
        from utils.quality.test_time_pick import score_clip_similarity

    n_ok = n_skip = n_bad_path = 0
    with out.open("w", encoding="utf-8") as fout:
        for row in _dict_rows(inp):
            path_s = _row_image(row)
            cap = first_str(row, ("caption", "text"))
            if not path_s:
                n_skip += 1
                continue
            img_path = resolve_image_path(path_s, image_root, inp.parent)
            rgb = _load_rgb_uint8(img_path) if img_path is not None else None
            if rgb is None:
                n_bad_path += 1
                continue
            row = dict(row)
            row["aesthetic_proxy"] = round(float(score_aesthetic_proxy(rgb)), 6)
            if args.with_phash:
                h = perceptual_hash(img_path)
                if h:
                    row["dedupe_phash"] = h
            if score_clip_similarity is not None:
                sims = score_clip_similarity([rgb], cap, device=device, model_id=str(args.clip_model)) if cap else []
                row["clip_sim"] = round(float(sims[0]), 6) if sims else 0.0
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            if args.max_rows and n_ok >= int(args.max_rows):
                break
    print(f"Wrote {n_ok} rows to {out} (skipped_no_path={n_skip}, missing_or_bad_image={n_bad_path})", file=sys.stderr)
    return 0


# --- image QC (shared by image-qc and gate) --------------------------------------------------


def _image_scores(manifest: Path, image_root: str, sample: int, counter: list[int]):
    """Yield ``(image_path, resolved, sharpness, contrast)``; ``counter[0]`` counts rows read."""
    import numpy as np
    from PIL import Image
    from utils.quality.finishing import rgb_to_luminance
    from utils.quality.test_time_pick import score_edge_sharpness

    root = Path(image_root) if image_root.strip() else None
    for row in _dict_rows(manifest):
        counter[0] += 1
        if sample and counter[0] > sample:
            break
        img_path = _row_image(row)
        path = resolve_image_path(img_path, root, manifest.parent)
        if path is None:
            continue
        try:
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
        yield img_path, path, float(score_edge_sharpness(rgb)), float(rgb_to_luminance(rgb).std())


def _qc_reasons(sharpness: float, contrast: float, min_sharpness: float, min_contrast: float) -> list[str]:
    reasons = []
    if min_sharpness > 0.0 and sharpness < min_sharpness:
        reasons.append(f"sharpness<{min_sharpness}")
    if min_contrast > 0.0 and contrast < min_contrast:
        reasons.append(f"contrast<{min_contrast}")
    return reasons


def _image_qc_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=_prog("image-qc"), description="Compute image sharpness/contrast for SDX JSONL manifests"
    )
    ap.add_argument("input", type=str, help="Path to JSONL manifest")
    ap.add_argument("--image-root", type=str, default="", help="Optional base dir to join with image paths")
    ap.add_argument("--sample", type=int, default=0, help="Process only first N rows (0=all)")
    ap.add_argument("--min-sharpness", type=float, default=0.0, help="Fail if sharpness < value (0=off)")
    ap.add_argument("--min-contrast", type=float, default=0.0, help="Fail if contrast < value (0=off)")
    ap.add_argument("--json-report", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 2
    processed = [0]
    results = []
    for img_path, path, sharpness, contrast in _image_scores(inp, args.image_root, args.sample, processed):
        reasons = _qc_reasons(sharpness, contrast, args.min_sharpness, args.min_contrast)
        results.append(
            {
                "image_path": img_path,
                "resolved_path": str(path),
                "sharpness": sharpness,
                "contrast": contrast,
                "ok": not reasons,
                "reasons": reasons,
            }
        )
    fail = any(not r["ok"] for r in results)
    report: dict[str, Any] = {
        "input": str(inp),
        "processed_rows": processed[0],
        "count": len(results),
        "min_sharpness": args.min_sharpness,
        "min_contrast": args.min_contrast,
        "fail": fail,
    }
    if results:
        for key in ("sharpness", "contrast"):
            vals = [r[key] for r in results]
            report[f"{key}_min"], report[f"{key}_max"] = min(vals), max(vals)
    if args.json_report:
        outp = Path(args.json_report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(
            json.dumps({"summary": report, "rows": results}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"imageqc: count={len(results)} fail={fail}")
    return 1 if fail else 0


# --- gate ------------------------------------------------------------------------------------


def _gate_caption_hygiene(manifest: Path, *, report_dups: bool, report_overlap: bool) -> dict[str, Any]:
    from sdx_native.text_hygiene import caption_fingerprint, normalize_caption_for_training, pos_neg_token_overlap

    dup_buckets: defaultdict[str, int] = defaultdict(int)
    overlap_rows = rows = 0
    for row in _dict_rows(manifest):
        cap = first_str(row, ("caption", "text"))
        if not cap:
            continue
        rows += 1
        if report_dups:
            dup_buckets[caption_fingerprint(cap, algorithm="sha256", normalize_first=True)] += 1
        neg = first_str(row, ("negative_caption", "negative_prompt", "negative_text"))
        if report_overlap and neg:
            inter, _u, j = pos_neg_token_overlap(
                normalize_caption_for_training(cap), normalize_caption_for_training(neg)
            )
            if inter > 0 and j > 0.0:
                overlap_rows += 1
    vals = list(dup_buckets.values())
    return {
        "rows_with_caption": rows,
        "dup_groups": sum(1 for v in vals if v > 1),
        "max_dup_count": max(vals, default=1),
        "pos_neg_overlap_rows": overlap_rows,
    }


def _gate_native(manifest: Path) -> dict[str, Any]:
    try:
        from utils.native import run_rust_jsonl_stats, run_rust_jsonl_validate
    except Exception:
        return {"available": False}
    out: dict[str, Any] = {"available": True}
    for key, fn in (("rust_validate", run_rust_jsonl_validate), ("rust_stats", run_rust_jsonl_stats)):
        try:
            out[f"{key}_rc"] = int(fn(manifest, timeout=600).returncode)
        except Exception as e:
            out[f"{key}_rc"] = None
            out[f"{key}_error"] = str(e)
    return out


def _gate_main(argv: list[str]) -> int:
    from utils.prompt.text import PromptLintOptions, lint_jsonl_path

    ap = argparse.ArgumentParser(
        prog=_prog("gate"),
        description="Manifest gate: prompt lint + caption hygiene + optional image QC + native stats. "
        "Reports by default; enable fail-fast thresholds via flags.",
    )
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--json-report", type=Path, default=None)
    ap.add_argument("--min-caption-len-chars", type=int, default=0)
    ap.add_argument("--max-caption-tokens", type=int, default=0)
    ap.add_argument("--fail-on-overlap", action="store_true")
    ap.add_argument("--report-dups", action="store_true")
    ap.add_argument("--report-overlap", action="store_true")
    ap.add_argument("--fail-on-dup-groups", type=int, default=0, help="Fail if dup_groups > N (0=off)")
    ap.add_argument(
        "--fail-on-caption-overlap-rows", type=int, default=0, help="Fail if pos/neg overlap rows > N (0=off)"
    )
    ap.add_argument("--image-qc", action="store_true", help="Run image sharpness/contrast QC (requires PIL).")
    ap.add_argument("--image-root", type=str, default="")
    ap.add_argument("--sample", type=int, default=0, help="Image QC: only first N rows (0=all)")
    ap.add_argument("--min-sharpness", type=float, default=0.0)
    ap.add_argument("--min-contrast", type=float, default=0.0)
    ap.add_argument("--native-stats", action="store_true", help="Try Rust validate/stats if available.")
    args = ap.parse_args(argv)

    manifest: Path = args.manifest
    if not manifest.is_file():
        print(f"Not a file: {manifest}", file=sys.stderr)
        return 2
    opts = PromptLintOptions(
        min_caption_len_chars=int(args.min_caption_len_chars),
        max_caption_tokens=int(args.max_caption_tokens),
        top_overlap_tokens=20,
        fail_on_overlap=bool(args.fail_on_overlap),
    )
    pl = dict(lint_jsonl_path(manifest, opts))
    ch = _gate_caption_hygiene(manifest, report_dups=args.report_dups, report_overlap=args.report_overlap)
    report: dict[str, Any] = {"manifest": str(manifest), "promptlint": pl, "caption_hygiene": ch}

    if args.image_qc:
        try:
            import PIL  # noqa: F401
        except Exception:
            report["image_qc"] = {"error": "PIL not installed"}
        else:
            processed = [0]
            scored = list(_image_scores(manifest, args.image_root or "", args.sample, processed))
            report["image_qc"] = {
                "processed_rows": processed[0],
                "images_scored": len(scored),
                "sharpness_min": min((s[2] for s in scored), default=0.0),
                "contrast_min": min((s[3] for s in scored), default=0.0),
                "fail": any(_qc_reasons(s[2], s[3], args.min_sharpness, args.min_contrast) for s in scored),
            }
    if args.native_stats:
        report["native"] = _gate_native(manifest)

    report["fail"] = bool(
        (args.fail_on_overlap and int(pl.get("pos_neg_overlap_rows", 0)) > 0)
        or (args.fail_on_dup_groups > 0 and ch["dup_groups"] > args.fail_on_dup_groups)
        or (args.fail_on_caption_overlap_rows > 0 and ch["pos_neg_overlap_rows"] > args.fail_on_caption_overlap_rows)
        or (isinstance(report.get("image_qc"), dict) and report["image_qc"].get("fail", False))
    )
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "manifest_gate:",
        f"fail={report['fail']}",
        f"rows_ok={pl.get('rows_ok')}",
        f"empty_caption_rows={pl.get('empty_caption_rows')}",
        f"pos_neg_overlap_rows={pl.get('pos_neg_overlap_rows')}",
        f"dup_groups={ch.get('dup_groups')}",
    )
    return 1 if report["fail"] else 0


# --- paths / merge / ar-tag / corpus ---------------------------------------------------------


def _paths_main(argv: list[str]) -> int:
    from utils.native import run_rust_dup_image_paths, run_rust_image_paths, rust_jsonl_tools_exe

    ap = argparse.ArgumentParser(
        prog=_prog("paths"),
        description="Extract image paths from SDX JSONL manifests (Rust sdx-jsonl-tools image-paths if built). "
        "Pipe to Zig sdx-pathstat for byte sizes.",
    )
    ap.add_argument("manifest", type=str, help="Path to .jsonl")
    ap.add_argument("--all-rows", action="store_true", help="Emit path per row (allow duplicates)")
    ap.add_argument("--sort", action="store_true", help="Sort paths lexicographically")
    ap.add_argument("--dup", action="store_true", help="Print duplicate path report (Rust dup-image-paths if built)")
    args = ap.parse_args(argv)

    mp = Path(args.manifest)
    if not mp.is_file():
        print(f"Not found: {mp}", file=sys.stderr)
        return 2
    exe = rust_jsonl_tools_exe()
    if args.dup and not exe:
        print("dup: build Rust sdx-jsonl-tools first.", file=sys.stderr)
        return 1
    if exe:
        r = (
            run_rust_dup_image_paths(mp, top=50)
            if args.dup
            else run_rust_image_paths(mp, all_rows=args.all_rows, sort=args.sort)
        )
        sys.stdout.write(r.stdout or "")
        if r.stderr:
            sys.stderr.write(r.stderr)
        return r.returncode

    paths = [p for p in map(_row_image, _dict_rows(mp)) if p]
    if not args.all_rows:
        paths = list(dict.fromkeys(paths))
    if args.sort:
        paths.sort()
    sys.stdout.write("".join(p + "\n" for p in paths))
    return 0


def _merge_main(argv: list[str]) -> int:
    from utils.native import merge_jsonl_files

    ap = argparse.ArgumentParser(
        prog=_prog("merge"),
        description="Merge JSONL manifests with dedupe (first wins); prefers Go sdx-manifest if built.",
    )
    ap.add_argument("-o", "--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--dedupe-key",
        default="image_path",
        help="JSON field for dedupe (default image_path; falls back to path/image in rows)",
    )
    ap.add_argument("inputs", nargs="+", help="Input .jsonl files (order preserved for first-wins)")
    args = ap.parse_args(argv)
    ins = [Path(p) for p in args.inputs]
    for p in ins:
        if not p.is_file():
            print(f"Not found: {p}", file=sys.stderr)
            return 1
    merge_jsonl_files(ins, Path(args.out), dedupe_key=args.dedupe_key, prefer_go=True)
    print(f"Wrote {args.out}")
    return 0


def _ar_tag_main(argv: list[str]) -> int:
    from utils.modeling.architecture import (
        ar_bridge_summary_lines,
        normalize_num_ar_blocks,
        read_num_ar_blocks_from_checkpoint,
        tag_manifest_row_ar,
    )

    p = argparse.ArgumentParser(
        prog=_prog("ar-tag"),
        description="Add num_ar_blocks (+ dit_num_ar_blocks, ar_regime) to JSONL from DiT ckpt or explicit value "
        "(see docs/MODEL_STACK.md).",
        epilog=ar_bridge_summary_lines(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dit-ckpt", type=Path, help="SDX train.py checkpoint (.pt) with config.num_ar_blocks")
    g.add_argument("--num-ar-blocks", type=int, help="Explicit 0, 2, or 4 (no checkpoint read)")
    p.add_argument("--manifest-jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="Replace existing valid AR fields on rows")
    args = p.parse_args(argv)

    if args.num_ar_blocks is not None:
        n = normalize_num_ar_blocks(args.num_ar_blocks)
        if n == -1:
            print("--num-ar-blocks must be 0, 2, or 4", file=sys.stderr)
            return 2
    else:
        n = read_num_ar_blocks_from_checkpoint(args.dit_ckpt)
        if n == -1:
            print(
                "Could not read num_ar_blocks from checkpoint (missing config?). Use --num-ar-blocks explicitly.",
                file=sys.stderr,
            )
            return 2
    if not args.manifest_jsonl.is_file():
        print(f"Not found: {args.manifest_jsonl}", file=sys.stderr)
        return 1
    rows = (tag_manifest_row_ar(r, n, overwrite=bool(args.overwrite)) for r in _dict_rows(args.manifest_jsonl))
    n_out = _write_jsonl(args.out, rows)
    print(f"[ar_tag_manifest] num_ar_blocks={n} rows={n_out} -> {args.out}")
    return 0


def _corpus_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog=_prog("corpus"), description="Corpus index build / retrieve")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build index JSONL from train manifest")
    b.add_argument("--jsonl", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--root", default="")
    b.add_argument("--max-rows", type=int, default=0)
    b.add_argument("--require-files", action="store_true")
    q = sub.add_parser("query", help="Retrieve refs for a prompt")
    q.add_argument("--index", required=True)
    q.add_argument("--prompt", required=True)
    q.add_argument("--top-k", type=int, default=6)
    q.add_argument("--source", default="", help="Filter: danbooru|e621|gelbooru|...")
    q.add_argument("--moodboard-out", default="")
    q.add_argument("--root", default="")
    args = p.parse_args(argv)

    from utils.generation.inventions import (
        build_corpus_index_from_jsonl,
        hits_to_moodboard_payload,
        retrieve_corpus_refs,
        save_corpus_index_meta,
    )

    if args.cmd == "build":
        idx = build_corpus_index_from_jsonl(
            args.jsonl,
            root=args.root or None,
            max_rows=int(args.max_rows),
            require_existing_file=bool(args.require_files),
        )
        print(f"indexed {len(idx)} rows -> {save_corpus_index_meta(idx, args.out)}")
        return 0
    idx = build_corpus_index_from_jsonl(args.index, root=args.root or None)
    hits = retrieve_corpus_refs(idx, args.prompt, top_k=int(args.top_k), source_filter=args.source)
    print(json.dumps([h.to_dict() for h in hits], indent=2))
    if args.moodboard_out:
        payload = hits_to_moodboard_payload(hits)
        Path(args.moodboard_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"moodboard -> {args.moodboard_out} ({len(payload.get('images') or [])} images)")
    return 0


# --- smoke dataset ---------------------------------------------------------------------------

_SMOKE_CAPTIONS = (
    "simple geometric pattern, flat colors",
    "soft gradient background, abstract",
    "noise texture, colorful",
    "minimal shapes, high contrast",
)


def _smoke_dataset_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=_prog("smoke-dataset"),
        description="Write synthetic images + .txt captions under <out>/train/ for train.py smoke tests "
        "(--data-path <out>; see docs/GETTING_STARTED.md).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory (created if missing).")
    p.add_argument("--count", type=int, default=4, help="Number of PNG files (default 4).")
    p.add_argument("--size", type=int, default=256, help="Square image side (default 256, match --image-size).")
    args = p.parse_args(argv)

    import numpy as np
    from PIL import Image

    train_dir = args.out / "train"  # Text2ImageDataset folder mode expects a subdir of images
    train_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(args.count))
    side = max(64, int(args.size))
    rng = np.random.default_rng(0)
    for i in range(n):
        base = np.zeros((side, side, 3), dtype=np.int16)
        base[..., i % 3] = 80 + (i * 37) % 160
        arr = np.clip(base + rng.integers(0, 40, size=(side, side, 3), dtype=np.uint8), 0, 255).astype(np.uint8)
        stem = f"smoke_{i:03d}"
        Image.fromarray(arr, mode="RGB").save(train_dir / f"{stem}.png")
        (train_dir / f"{stem}.txt").write_text(_SMOKE_CAPTIONS[i % len(_SMOKE_CAPTIONS)] + "\n", encoding="utf-8")
    print(f"[make_smoke_dataset] wrote {n} PNG + caption pairs under {train_dir.resolve()}")
    print("Next: see docs/GETTING_STARTED.md for a minimal train.py command.")
    return 0


# --- Hugging Face export ---------------------------------------------------------------------


def _to_pil(image_val: Any):
    """HF Image feature, PIL, numpy, or dict with bytes/path."""
    from PIL import Image

    if image_val is None:
        return None
    if hasattr(image_val, "save"):
        return image_val
    if isinstance(image_val, dict):
        if image_val.get("bytes") is not None:
            return Image.open(io.BytesIO(image_val["bytes"])).convert("RGB")
        if image_val.get("path"):
            return Image.open(image_val["path"]).convert("RGB")
    try:
        import numpy as np

        if isinstance(image_val, np.ndarray):
            arr = image_val.astype("uint8")
            return Image.fromarray(arr, mode="L").convert("RGB") if arr.ndim == 2 else Image.fromarray(arr, mode="RGB")
    except Exception:
        pass
    return None


def _write_image_fast(image_val: Any, ipath: Path, img_ext: str) -> bool:
    """Copy encoded bytes / source file straight to disk when the format already matches."""
    if not isinstance(image_val, dict):
        return False
    b = image_val.get("bytes")
    if isinstance(b, (bytes, bytearray)) and len(b) > 16:
        b = bytes(b)
        fmt = (
            "jpg"
            if b[:2] == b"\xff\xd8"
            else "png"
            if b[:8] == b"\x89PNG\r\n\x1a\n"
            else "webp"
            if b[:4] == b"RIFF" and b[8:12] == b"WEBP"
            else None
        )
        if fmt == img_ext:
            ipath.write_bytes(b)
            return True
    p = image_val.get("path")
    if p and Path(str(p)).suffix.lower().lstrip(".").replace("jpeg", "jpg") == img_ext:
        try:
            shutil.copyfile(str(p), ipath)
            return True
        except OSError:
            pass
    return False


def _prefetch_rows(row_iter, bufsize: int):
    q: Queue[Any] = Queue(maxsize=max(32, bufsize))
    stop = object()

    def _reader() -> None:
        try:
            for row in row_iter:
                q.put(row)
        finally:
            q.put(stop)

    threading.Thread(target=_reader, daemon=True).start()
    while (item := q.get()) is not stop:
        yield item


def _hf_caption(row: dict[str, Any], caption_field: str, tag_join: str) -> str:
    for field in (caption_field, "text", "tag_string", "tags"):
        v = row.get(field)
        if v is None:
            continue
        cap = (
            tag_join.join(str(x).strip() for x in v if str(x).strip())
            if isinstance(v, (list, tuple))
            else str(v).strip()
        )
        if cap:
            return cap
    return ""


def _prepare_for_save(pil, img_ext: str):
    """JPEG cannot store alpha; flatten RGBA/P/LA to RGB."""
    if img_ext != "jpg" or pil.mode == "RGB":
        return pil
    if pil.mode in ("RGBA", "LA"):
        from PIL import Image

        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[-1])
        return bg
    return pil.convert("RGB")


def _save_sample(
    row: dict[str, Any],
    *,
    stem: str,
    img_dir: Path,
    img_ext: str,
    save_kw: dict[str, Any],
    caption_field: str,
    image_field: str,
    tag_join: str,
    skip_reencode: bool = False,
) -> str | None:
    from data.manifest_utils import manifest_row_text
    from utils.safety.minor_guard import reasons as minor_guard_reasons

    cap = _hf_caption(row, caption_field, tag_join)
    if not cap or minor_guard_reasons(f"{cap}, {manifest_row_text(row)}", training=True):
        return None
    image_val = row.get(image_field)
    ipath = img_dir / f"{stem}.{img_ext}"
    if not (skip_reencode and _write_image_fast(image_val, ipath, img_ext)):
        pil = _to_pil(image_val)
        if pil is None:
            return None
        _prepare_for_save(pil, img_ext).save(ipath, **save_kw)
    return json.dumps({"image_path": f"images/{stem}.{img_ext}", "caption": cap}, ensure_ascii=False) + "\n"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "yes")


def _hf_export_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=_prog("hf-export"),
        description="HF Dataset -> SDX JSONL + image files (needs an image column; see docs/GUIDES.md). "
        "Use --streaming with --max-samples for huge datasets.",
    )
    p.add_argument("--dataset", type=str, required=True, help="HF dataset id, e.g. org/danbooru-export")
    p.add_argument(
        "--config", type=str, default=None, help="Optional dataset config name (second arg to load_dataset)."
    )
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", type=str, default=None, help="Git revision / branch / commit.")
    p.add_argument("--image-field", type=str, default="image", help="Column with image (PIL/bytes).")
    p.add_argument("--caption-field", type=str, default="tag_string", help="Column for caption/tags.")
    p.add_argument(
        "--caption-tag-join",
        type=str,
        default=", ",
        help="If caption field is a list of tags, join with this string (default ', ').",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output root; images/ + manifest.jsonl")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=0, help="Stop after N rows (0 = no limit).")
    p.add_argument("--streaming", action="store_true", help="IterableDataset mode (recommended for huge data).")
    p.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle seed when streaming (buffered shuffle).")
    p.add_argument("--start-index", type=int, default=0, help="Skip first N rows (non-streaming only).")
    p.add_argument("--list-columns", action="store_true", help="Print first row keys and exit.")
    p.add_argument("--image-format", default="png", choices=("png", "jpg", "jpeg", "webp"), help="Saved image format.")
    p.add_argument("--workers", type=int, default=0, help="Parallel image writers (0=auto from SDX_HF_EXPORT_WORKERS).")
    p.add_argument("--jpeg-quality", type=int, default=0, help="JPEG quality 1-95 (0=SDX_HF_JPEG_QUALITY or 95).")
    p.add_argument("--turbo", action="store_true", help="Fast JPEG (no optimize, lower quality).")
    args = p.parse_args(argv)

    turbo = args.turbo or _env_flag("SDX_HF_TURBO", "")
    workers = args.workers or int(os.environ.get("SDX_HF_EXPORT_WORKERS", "0") or 0)
    if workers < 1:
        workers = 48 if turbo else 1
    jpg_q = args.jpeg_quality or int(os.environ.get("SDX_HF_JPEG_QUALITY", "0") or 0) or (78 if turbo else 95)
    skip_reencode = _env_flag("SDX_HF_SKIP_REENCODE", "1" if turbo else "0")
    prefetch = int(os.environ.get("SDX_HF_PREFETCH", "0") or 0)
    if prefetch < 1 and turbo:
        prefetch = max(workers * 12, 384)

    img_ext = args.image_format.lower().replace("jpeg", "jpg")
    if img_ext == "jpg":
        save_kw = {
            "format": "JPEG",
            "quality": max(1, min(95, int(jpg_q))),
            "optimize": not turbo,
            "subsampling": 2 if turbo else 0,
        }
    elif img_ext == "webp":
        save_kw = {"format": "WEBP", "quality": 92}
    else:
        save_kw = {"format": "PNG"}

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets", file=sys.stderr)
        return 1

    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / args.manifest_name
    ld_args = (args.dataset,) if args.config is None else (args.dataset, args.config)
    ld_kw: dict[str, Any] = {"revision": args.revision} if args.revision else {}

    if args.streaming:
        split_ds = load_dataset(*ld_args, split=args.split, streaming=True, **ld_kw)
        if args.shuffle_seed is not None:
            split_ds = split_ds.shuffle(seed=int(args.shuffle_seed), buffer_size=10_000)
        if args.list_columns:
            row = next(iter(split_ds))
            print("Columns:", list(row.keys()) if isinstance(row, dict) else row)
            return 0
        base_rows = iter(split_ds)
    else:
        raw = load_dataset(*ld_args, **ld_kw)
        split_ds = raw[args.split] if isinstance(raw, dict) else raw
        if args.start_index:
            split_ds = split_ds.select(range(int(args.start_index), len(split_ds)))
        if args.list_columns:
            print("Columns:", split_ds.column_names)
            row = split_ds[0]
            print("First row keys:", list(row.keys()) if isinstance(row, dict) else type(row))
            return 0
        base_rows = (split_ds[i] for i in range(len(split_ds)))

    rows = _prefetch_rows(base_rows, prefetch) if workers > 1 and prefetch > 0 else base_rows
    max_samples = int(args.max_samples)
    flush_every = max(1, int(os.environ.get("SDX_HF_MANIFEST_FLUSH_EVERY", "128") or 128))
    save = dict(
        img_dir=img_dir,
        img_ext=img_ext,
        save_kw=save_kw,
        caption_field=args.caption_field,
        image_field=args.image_field,
        tag_join=args.caption_tag_join,
        skip_reencode=skip_reencode,
    )
    n_written = 0

    def _emit(mf, line: str | None) -> None:
        nonlocal n_written
        if not line:
            return
        mf.write(line)
        n_written += 1
        if n_written % flush_every == 0:
            mf.flush()
            print(f"  wrote {n_written} samples …", flush=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        if workers <= 1:
            for i, row in enumerate(rows):
                if max_samples and n_written >= max_samples:
                    break
                _emit(mf, _save_sample(dict(row), stem=f"{i:08d}", **save))
        else:
            print(
                f"  turbo export: {workers} workers, prefetch={prefetch}, "
                f"jpeg_q={save_kw.get('quality', '?')}, skip_reencode={skip_reencode}",
                flush=True,
            )
            max_in_flight = workers * 8
            with ThreadPoolExecutor(max_workers=workers) as ex:
                pending: set = set()
                for i, row in enumerate(rows):
                    if max_samples and i >= max_samples:
                        break
                    pending.add(ex.submit(_save_sample, dict(row), stem=f"{i:08d}", **save))
                    if len(pending) >= max_in_flight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            _emit(mf, fut.result())
                for fut in pending:
                    _emit(mf, fut.result())

    print(f"Done: {n_written} rows -> {manifest_path}")
    print(f"Train with: python train.py --manifest-jsonl {manifest_path} --data-path {args.out_dir} ...")
    return 0


# DataLoader workers > 0 often deadlocks on Windows; default to 0 there.
_DEFAULT_TRAIN_WORKERS = 0 if sys.platform == "win32" else 2


def _hf_train_main(argv: list[str]) -> int:
    if "--" in argv:
        i = argv.index("--")
        argv, train_argv = argv[:i], argv[i + 1 :]
    else:
        train_argv = []
    p = argparse.ArgumentParser(
        prog=_prog("hf-train"),
        description="HF export + basic DiT training (extra train.py args after --). --demo uses synthetic images.",
    )
    p.add_argument("--demo", action="store_true", help="Skip HF: synthetic smoke dataset + train (no download).")
    p.add_argument("--out-dir", type=Path, default=Path("data/hf_basic_run"))
    p.add_argument("--results-dir", type=Path, default=Path("results/hf_basic_run"))
    p.add_argument("--dataset", type=str, default="", help="HF dataset id (required unless --demo).")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--image-field", type=str, default="image")
    p.add_argument("--caption-field", type=str, default="tag_string")
    p.add_argument("--caption-tag-join", type=str, default=", ")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="Load the full split into memory instead of streaming (can use a lot of RAM).",
    )
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--model", type=str, default="DiT-B/2-Text")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--global-batch-size", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=100, help="Passed to train.py (ignored if --dry-run).")
    p.add_argument("--dry-run", action="store_true", help="Only 1 training step.")
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"DataLoader workers for train.py (default: {_DEFAULT_TRAIN_WORKERS} on this OS).",
    )
    p.add_argument(
        "--no-xformers", action="store_true", help="Pass --no-xformers to train.py (helps some Windows/CUDA setups)."
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir
    if args.demo:
        rc = _smoke_dataset_main(["--out", str(out_dir), "--count", "8"])
        data_args = ["--data-path", str(out_dir)]
    else:
        if not args.dataset.strip():
            print("Error: pass --dataset YOUR_ORG/name or use --demo", file=sys.stderr)
            return 1
        exp = [
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--image-field",
            args.image_field,
            "--caption-field",
            args.caption_field,
            "--caption-tag-join",
            args.caption_tag_join,
            "--out-dir",
            str(out_dir),
            "--manifest-name",
            args.manifest_name,
            "--max-samples",
            str(args.max_samples),
        ]
        if args.config:
            exp += ["--config", args.config]
        if args.revision:
            exp += ["--revision", args.revision]
        if not args.no_streaming:
            exp.append("--streaming")
        if args.shuffle_seed is not None:
            exp += ["--shuffle-seed", str(args.shuffle_seed)]
        print("[hf_download_and_train] Export:", " ".join(exp))
        rc = _hf_export_main(exp)
        data_args = ["--manifest-jsonl", str(out_dir / args.manifest_name), "--data-path", str(out_dir)]
    if rc:
        return rc

    nw = _DEFAULT_TRAIN_WORKERS if args.num_workers is None else args.num_workers
    train_cmd = [
        sys.executable,
        str(_ROOT / "train.py"),
        *data_args,
        "--results-dir",
        str(args.results_dir),
        "--model",
        args.model,
        "--image-size",
        str(args.image_size),
        "--global-batch-size",
        str(args.global_batch_size),
        "--no-compile",
        "--num-workers",
        str(nw),
        *(["--dry-run"] if args.dry_run else ["--max-steps", str(args.max_steps)]),
        *(["--no-xformers"] if args.no_xformers else []),
        *train_argv,
    ]
    print("[hf_download_and_train] Train:", " ".join(train_cmd))
    return subprocess.run(train_cmd, cwd=_ROOT).returncode


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "quality": _quality_main,
    "enrich": _enrich_main,
    "gate": _gate_main,
    "image-qc": _image_qc_main,
    "paths": _paths_main,
    "merge": _merge_main,
    "ar-tag": _ar_tag_main,
    "corpus": _corpus_main,
    "smoke-dataset": _smoke_dataset_main,
    "hf-export": _hf_export_main,
    "hf-train": _hf_train_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
