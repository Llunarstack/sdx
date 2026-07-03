#!/usr/bin/env python3
"""Build per-artist and per-style subset manifests for LoRA bank training.

    python setup/build_lora_subsets.py \\
        --manifest /workspace/data/enriched/manifest.jsonl \\
        --data-root /workspace/data \\
        --out /workspace/data/lora_subsets
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.defaults.art_mediums import MEDIUM_SPECS  # noqa: E402
from utils.lora.lora_bank import slugify_lora_key  # noqa: E402

_STYLE_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("anime", ("anime", "manga", "cel shading", "toon shading", "1girl", "1boy", "solo")),
    ("digital_painting", ("digital painting", "digital art", "photoshop", "procreate", "illustration")),
    ("realistic", ("photorealistic", "photo realistic", "realistic", "photograph", "dslr")),
    ("3d_render", ("3d render", "blender", "octane", "unreal engine", "cgi", "ray tracing")),
    ("pixel_art", ("pixel art", "pixelated", "8-bit", "16-bit")),
    ("concept_art", ("concept art", "matte painting", "environment concept")),
]

for pack in MEDIUM_SPECS:
    if pack.id not in {b[0] for b in _STYLE_BUCKETS}:
        kws = tuple(pack.keywords or ())[:6]
        if kws:
            _STYLE_BUCKETS.append((pack.id, kws))


def _caption_text(row: dict) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in ("caption", "tags", "booru_caption", "scene_summary")
    ).lower()


def _classify_style(row: dict) -> str | None:
    text = _caption_text(row)
    best_id, best_score = None, 0
    for style_id, keywords in _STYLE_BUCKETS:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_id = style_id
    return best_id if best_score > 0 else None


def build_subsets(
    manifest: Path,
    out_dir: Path,
    *,
    min_artist_samples: int = 150,
    max_artists: int = 64,
    min_style_samples: int = 200,
) -> dict:
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    artist_rows: dict[str, list[dict]] = defaultdict(list)
    style_rows: dict[str, list[dict]] = defaultdict(list)
    artist_counts: Counter[str] = Counter()

    for row in rows:
        artists = list(row.get("artist_tags") or [])
        if not artists:
            for tag in str(row.get("caption") or "").split(","):
                t = tag.strip()
                if t.startswith("artist:"):
                    artists.append(t.split(":", 1)[-1].strip())
        for a in artists:
            slug = slugify_lora_key(str(a))
            if slug:
                artist_rows[slug].append(row)
                artist_counts[slug] += 1

        style = _classify_style(row)
        if style:
            style_rows[style].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"artists": {}, "styles": {}}

    top_artists = [k for k, c in artist_counts.most_common(max_artists) if c >= min_artist_samples]
    for slug in top_artists:
        subset = artist_rows[slug]
        rel = out_dir / "artist" / slug / "manifest.jsonl"
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8")
        meta["artists"][slug] = {"manifest": str(rel), "count": len(subset)}

    for style_id, subset in style_rows.items():
        if len(subset) < min_style_samples:
            continue
        rel = out_dir / "style" / style_id / "manifest.jsonl"
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8")
        meta["styles"][style_id] = {"manifest": str(rel), "count": len(subset)}

    meta_path = out_dir / "subsets.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"LoRA subsets: {len(meta['artists'])} artists, {len(meta['styles'])} styles -> {out_dir}",
        flush=True,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build LoRA training subset manifests.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=None, help="Unused; kept for CLI symmetry.")
    p.add_argument("--out", required=True)
    p.add_argument("--min-artist-samples", type=int, default=150)
    p.add_argument("--max-artists", type=int, default=64)
    p.add_argument("--min-style-samples", type=int, default=200)
    args = p.parse_args(argv)
    build_subsets(
        Path(args.manifest),
        Path(args.out),
        min_artist_samples=args.min_artist_samples,
        max_artists=args.max_artists,
        min_style_samples=args.min_style_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
