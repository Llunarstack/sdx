#!/usr/bin/env python3
"""Manifest prep for training: merge, sanitize, clean media, WD-tag, enrich captions, control maps.

    python setup/manifests.py merge --data-root /workspace/data --out /workspace/data/combined/manifest.jsonl
    python setup/manifests.py sanitize --manifest M --data-root D --backup --verify-images
    python setup/manifests.py cleanup-media --data-root /workspace/data --rewrite-manifests --backup
    python setup/manifests.py wd-tag --manifest M --data-root D --out /workspace/data/tagged/manifest.jsonl
    python setup/manifests.py enrich --manifest M --data-root D --out /workspace/data/enriched/manifest.jsonl
    python setup/manifests.py control-maps --manifest M --data-root D --control-type canny --out OUT

Per-site manifests use paths relative to their site folder; ``merge`` rewrites ``image_path`` to
``<site>/images/<file>`` so one ``--data-path`` root serves every site.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.manifest_utils import manifest_row_text, resolve_image_path  # noqa: E402
from scripts.scrape import (  # noqa: E402
    DEFAULT_SCRAPE_SITES,
    SPLITTABLE_SOURCE_EXTS,
    TRAINABLE_IMAGE_EXTS,
    validate_trainable_image,
)
from utils.captioning import WDTagger, merge_wd_tags_into_row  # noqa: E402
from utils.safety.minor_guard import reasons as minor_guard_reasons  # noqa: E402


def _read_rows(manifest: Path, max_rows: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:max_rows] if max_rows > 0 else rows


# --- merge ----------------------------------------------------------------------


def merge_manifests(data_root: Path, out_path: Path, sites: list[str]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_md5: set[str] = set()
    written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for site in sites:
            manifest = data_root / site / "manifest.jsonl"
            if not manifest.is_file():
                print(f"[skip] {site}: no {manifest}")
                continue
            with manifest.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    md5 = row.get("md5", "")
                    if md5:
                        if md5 in seen_md5:
                            continue
                        seen_md5.add(md5)
                    rel = row.get("image_path", "")
                    if rel and not str(rel).startswith(f"{site}/"):
                        row["image_path"] = f"{site}/{rel}".replace("\\", "/")
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
            print(f"[ok] {site}: merged into {out_path.name}")
    return written


def _merge(args: argparse.Namespace) -> int:
    n = merge_manifests(Path(args.data_root), Path(args.out), list(args.sites))
    print(f"Wrote {n:,} rows -> {args.out}")
    return 0


# --- sanitize / cleanup-media ---------------------------------------------------


def filter_manifest(manifest: Path, root: Path, *, verify: bool, backup: bool) -> tuple[int, int]:
    """Drop rows whose image is missing, not a trainable extension, or (``verify``) not a readable raster."""
    if not manifest.is_file():
        return 0, 0
    if backup:
        shutil.copy(manifest, manifest.with_suffix(manifest.suffix + ".bak"))
    kept: list[str] = []
    removed = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        fp = root / str(d.get("image_path") or d.get("path") or d.get("image") or "")
        if fp.is_file() and fp.suffix.lower() in TRAINABLE_IMAGE_EXTS and (not verify or validate_trainable_image(fp)):
            kept.append(line)
        else:
            removed += 1
    manifest.write_text("".join(f"{k}\n" for k in kept), encoding="utf-8")
    return len(kept), removed


def _sanitize(args: argparse.Namespace) -> int:
    kept, removed = filter_manifest(
        Path(args.manifest), Path(args.data_root), verify=args.verify_images, backup=args.backup
    )
    print(f"kept {kept} rows, removed {removed} -> {args.manifest}")
    return 0


def cleanup_images_dir(images_dir: Path, *, drop_raw_media: bool, drop_invalid: bool) -> dict[str, int]:
    """Delete partial downloads, blocked extensions, raw animations already split into frames, corrupt images."""
    counts = dict.fromkeys(("part", "blocked_ext", "raw_with_frames", "raw_orphan", "invalid_image", "kept"), 0)
    if not images_dir.is_dir():
        return counts
    frame_parents = {p.stem.rsplit("_f", 1)[0] for p in images_dir.glob("*_f*.jpg")}
    for path in list(images_dir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in {".part", ".tmp", ".crdownload"}:
            path.unlink(missing_ok=True)
            counts["part"] += 1
        elif ext in SPLITTABLE_SOURCE_EXTS:
            has_frames = path.stem in frame_parents
            if has_frames or drop_raw_media:
                path.unlink(missing_ok=True)
                counts["raw_with_frames" if has_frames else "raw_orphan"] += 1
        elif ext in TRAINABLE_IMAGE_EXTS:
            if drop_invalid and not validate_trainable_image(path):
                path.unlink(missing_ok=True)
                counts["invalid_image"] += 1
            else:
                counts["kept"] += 1
        else:  # blocked download extensions and anything unknown
            path.unlink(missing_ok=True)
            counts["blocked_ext"] += 1
    return counts


def _cleanup_media(args: argparse.Namespace) -> int:
    root = Path(args.data_root)
    totals: dict[str, int] = {}
    for site in args.sites:
        site_root = root / site
        stats = cleanup_images_dir(site_root / "images", drop_raw_media=args.drop_raw_media, drop_invalid=True)
        print(f"[{site}] images: {stats}")
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        if args.rewrite_manifests:
            kept, removed = filter_manifest(site_root / "manifest.jsonl", site_root, verify=True, backup=args.backup)
            print(f"[{site}] manifest: kept {kept}, removed {removed}")
    print(f"total: {totals}")
    return 0


# --- wd-tag (WD EVA02 tags + minor screen) ---------------------------------------
# Identity tags (character, artist, copyright) stay from the booru API; WD adds hair, pose, clothing, scene.
# Run **before** caption enrichment. Minor screen (always on): a row is dropped when its WD tags at
# ``SCREEN_THRESHOLD`` or its manifest text trip utils.safety.minor_guard's training rule (any youth term,
# any rating). It catches images whose booru tags omit youth tags (Thiel, Stanford Internet Observatory,
# 2023, on CSAM in LAION-5B).

# Below the 0.35 caption threshold: a missed youth tag costs more than a dropped adult row.
SCREEN_THRESHOLD = 0.2


def _resolve_image(data_root: Path, row: dict) -> Path | None:
    rel = str(row.get("image_path") or row.get("path") or "").strip()
    site = str(row.get("site") or "").strip()
    found = resolve_image_path(rel, data_root)
    if found is None and rel and site:
        cand = data_root / site / "images" / Path(rel).name
        found = cand if cand.is_file() else None
    return found


def tag_manifest(
    manifest: Path,
    data_root: Path,
    out: Path,
    *,
    model_dir: Path | None = None,
    threshold: float = 0.35,
    max_rows: int = 0,
    skip_complete: bool = True,
) -> int:
    rows = _read_rows(manifest, max_rows)
    tagger = WDTagger(model_dir, threshold=threshold)
    youth_vocab = [n for n in tagger.tag_names if minor_guard_reasons(n, training=True)]
    if not youth_vocab:
        raise SystemExit(
            f"{tagger.model_dir / 'selected_tags.csv'} has no youth tags; the minor screen would be a no-op."
        )
    print(f"WD minor screen: {len(youth_vocab)} youth tags in the vocabulary ({', '.join(youth_vocab[:8])}, ...)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    written = errors = dropped = 0

    with tmp.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            done = skip_complete and "wd_tagger" in [str(s) for s in (row.get("tag_sources") or [])]
            img = None if done else _resolve_image(data_root, row)
            if not done and img is None:
                errors += 1
            if img is not None:
                try:
                    tags = tagger.scores(img)
                except Exception as exc:
                    # An image the screen cannot see is not written (fail closed).
                    print(f"  row {i}: WD tag failed, row dropped ({exc})", file=sys.stderr)
                    errors += 1
                    continue
                if minor_guard_reasons(", ".join(t.name for t in tags if t.score >= SCREEN_THRESHOLD), training=True):
                    dropped += 1
                    continue
                row = merge_wd_tags_into_row(row, [t.name for t in tagger.select(tags)])
            if minor_guard_reasons(manifest_row_text(row), training=True):
                dropped += 1
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if i % 500 == 0:
                print(f"  tagged {i}/{len(rows)}...", flush=True)

    tmp.replace(out)
    print(f"WD-tagged {written} rows -> {out} ({errors} errors, {dropped} dropped by the minor screen)")
    return 0 if written else 1


def _wd_tag(args: argparse.Namespace) -> int:
    return tag_manifest(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.out),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        threshold=args.threshold,
        max_rows=args.max_rows,
        skip_complete=not args.force,
    )


# --- enrich (multi-source caption profiling) --------------------------------------
# Default (SDX_PROMPT_RESEARCH=1): VLM + local RAG + Creative RAG (Qwen) write detailed diffusion prompts with
# booru identity tags merged in. Fast path: --booru-only or SDX_PROMPT_RESEARCH=0 restructures API tags only.
# No API keys needed — SauceNAO/TinEye use the same web upload as a browser.


def _row_needs_enrich(row: dict, *, use_prompt_research: bool) -> bool:
    if use_prompt_research:
        sources = [str(s) for s in (row.get("tag_sources") or [])]
        return not any(x in s for x in ("creative_rag", "vlm_uncensored", "rag_fallback") for s in sources)
    return not row.get("scene_summary")


def _merge_research_row(row: dict, researched) -> dict:
    """Fuse booru identity tags with the VLM+RAG+LLM diffusion prompt."""
    merged = dict(row)
    patch = researched.to_manifest_patch()
    cap = str(patch.get("caption") or "").strip()
    identity: list[str] = []
    for key in ("character_tags", "copyright_tags", "artist_tags"):
        for t in row.get(key) or []:
            t = str(t).strip().replace("_", " ")
            if t and t.lower() not in cap.lower():
                identity.append(t)
    if identity:
        patch["caption"] = ", ".join(identity) + (", " + cap if cap else "")
    if row.get("caption"):
        patch["booru_caption"] = row["caption"]
    sources = [str(s) for s in (row.get("tag_sources") or [])]
    sources.extend(str(s) for s in (patch.get("tag_sources") or []))
    patch["tag_sources"] = list(dict.fromkeys(sources))
    merged.update(patch)
    return merged


def enrich_manifest(
    manifest: Path,
    data_root: Path,
    out: Path,
    *,
    workers: int = 4,
    max_rows: int = 0,
    use_vlm: bool = True,
    use_reverse: bool = True,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    use_prompt_research: bool = False,
    rag_corpus: Path | None = None,
    skip_complete: bool = True,
) -> int:
    from utils.captioning import profile_from_manifest_row
    from utils.captioning import research_prompt_for_image

    rows = _read_rows(manifest, max_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    reverse_on = use_reverse and (use_saucenao or use_tineye)
    corpus = Path(rag_corpus) if rag_corpus and Path(rag_corpus).is_file() else None
    if use_prompt_research:
        workers = 1  # VLM + LLM are GPU-serial; parallel workers OOM the pod

    def _one(row: dict) -> dict:
        if skip_complete and not _row_needs_enrich(row, use_prompt_research=use_prompt_research):
            return row
        try:
            if use_prompt_research:
                rel = row.get("image_path") or ""
                img = data_root / rel if rel else Path("_missing")
                if not img.is_file():
                    img = Path(rel) if rel else Path("_missing")
                if img.is_file():
                    researched = research_prompt_for_image(
                        img, rag_corpus=corpus, seed_prompt=str(row.get("caption") or ""), use_rag=corpus is not None
                    )
                    return _merge_research_row(row, researched)
            prof = profile_from_manifest_row(
                row,
                data_root,
                use_vlm=use_vlm,
                use_reverse_search=reverse_on,
                use_saucenao=use_saucenao,
                use_tineye=use_tineye,
            )
            merged = dict(row)
            merged.update(prof.to_manifest_patch())
            return merged
        except Exception:
            return row

    written = 0
    with open(tmp, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(_one, rows):  # map keeps input order
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"enrichment produced 0 rows — not overwriting {out}")
    tmp.replace(out)
    return written


def _enrich(args: argparse.Namespace) -> int:
    if args.booru_only:
        use_prompt_research = False
    elif args.prompt_research is not None:
        use_prompt_research = bool(args.prompt_research)
    else:
        use_prompt_research = os.environ.get("SDX_PROMPT_RESEARCH", "1") == "1"

    rag = Path(args.rag_corpus) if args.rag_corpus else None
    if use_prompt_research and rag is None:
        default_rag = Path(os.environ.get("SDX_RAG_CORPUS", Path(args.data_root) / "rag_corpus.jsonl"))
        rag = default_rag if default_rag.is_file() else None

    n = enrich_manifest(
        Path(args.manifest),
        Path(args.data_root),
        Path(args.out),
        workers=args.workers,
        max_rows=args.max_rows,
        use_vlm=not args.no_vlm and not use_prompt_research,
        use_reverse=not args.no_reverse_search and not use_prompt_research,
        use_saucenao=not args.no_saucenao,
        use_tineye=not args.no_tineye,
        use_prompt_research=use_prompt_research,
        rag_corpus=rag,
        skip_complete=not args.force,
    )
    mode = "VLM+RAG+LLM prompt research" if use_prompt_research else "booru metadata"
    print(f"Wrote {n:,} rows -> {args.out} ({mode})")
    print("Point training at enriched manifest; scene_summary feeds RAG via `setup/indexes.py rag-corpus`")
    return 0


# --- control-maps (ControlNet training pairs) ---------------------------------------
# Writes ``controls/<type>/<md5>.png`` and a manifest where every row has ``control_image`` + ``control_type``.
# Train with: python train.py --manifest-jsonl <out> --control-cond-dim 1 --control-num-types 9 ...


def _extract_one(data_root: Path, row: dict, *, control_type: str, controls_dir: Path) -> dict | None:
    img = resolve_image_path(row.get("image_path") or row.get("path") or "", data_root)
    if img is None:
        return None
    md5 = row.get("md5") or img.stem.split("_f")[0]
    dest = controls_dir / control_type / f"{md5}.png"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from utils.modeling.hf_loaders import extract_control_maps_batch, extract_pil_proxy

            maps = extract_control_maps_batch(str(img), dest.parent, types=(control_type,))
            if control_type not in maps:
                if not extract_pil_proxy(str(img), str(dest), control_type):
                    return None
            else:
                src = Path(maps[control_type])  # batch writes stem_type.png — rename to md5
                if src != dest and src.is_file():
                    src.replace(dest)
        except Exception:
            try:
                import cv2
                import numpy as np
                from PIL import Image

                with Image.open(img) as im:
                    gray = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
                cv2.imwrite(str(dest), cv2.Canny(gray, 80, 160))
            except Exception:
                return None
    out_row = dict(row)
    out_row["control_image"] = dest.relative_to(data_root).as_posix()
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
    rows_in = _read_rows(manifest, max_rows)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_manifest, "w", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(_extract_one, data_root, row, control_type=control_type, controls_dir=controls_dir)
            for row in rows_in
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1
    return written


def _control_maps(args: argparse.Namespace) -> int:
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


# --- CLI ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("merge", help="Merge booru site manifests (dedupe by md5)")
    p.add_argument("--data-root", required=True, help="Base dir containing per-site folders.")
    p.add_argument("--out", required=True, help="Output combined manifest.jsonl path.")
    p.add_argument("--sites", nargs="*", default=list(DEFAULT_SCRAPE_SITES), help="Sites to include.")
    p.set_defaults(fn=_merge)

    p = sub.add_parser("sanitize", help="Remove broken/missing rows from a training manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--backup", action="store_true", help="Write manifest.bak before editing.")
    p.add_argument(
        "--verify-images", action="store_true", help="PIL-verify each image (slower; drops corrupt PNG/JPEG)."
    )
    p.set_defaults(fn=_sanitize)

    p = sub.add_parser("cleanup-media", help="Clean scraped booru media folders")
    p.add_argument("--data-root", required=True)
    p.add_argument("--sites", nargs="*", default=list(DEFAULT_SCRAPE_SITES))
    p.add_argument(
        "--drop-raw-media", action="store_true", help="Delete all gif/mp4/webm even when frame JPEGs are missing."
    )
    p.add_argument("--rewrite-manifests", action="store_true")
    p.add_argument("--backup", action="store_true")
    p.set_defaults(fn=_cleanup_media)

    p = sub.add_parser("wd-tag", help="WD tagger manifest enrichment (with mandatory minor screen)")
    p.add_argument("--manifest", required=True, help="Input manifest.jsonl")
    p.add_argument("--data-root", required=True, help="Dataset root for image paths")
    p.add_argument("--out", required=True, help="Output manifest.jsonl")
    p.add_argument("--model-dir", default=None, help="WD-EVA02-Large-Tagger folder")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Re-tag rows that already have wd_tagger")
    p.set_defaults(fn=_wd_tag)

    p = sub.add_parser("enrich", help="Multi-source caption enrichment (booru + SauceNAO + VLM)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--no-reverse-search", action="store_true")
    p.add_argument("--no-saucenao", action="store_true")
    p.add_argument("--no-tineye", action="store_true")
    p.add_argument(
        "--prompt-research",
        action="store_true",
        default=None,
        help="VLM + local RAG + Creative RAG (Qwen) for detailed training captions (default: SDX_PROMPT_RESEARCH=1).",
    )
    p.add_argument("--booru-only", action="store_true", help="Fast path: booru tag restructuring only.")
    p.add_argument("--rag-corpus", default="", help="rag_corpus.jsonl for --prompt-research")
    p.add_argument("--force", action="store_true", help="Re-profile even when tags look complete.")
    p.set_defaults(fn=_enrich)

    p = sub.add_parser("control-maps", help="Build control-paired manifest for ControlNet training")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--control-type", default="canny", choices=["canny", "softedge", "hed", "depth", "normals"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all).")
    p.set_defaults(fn=_control_maps)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
