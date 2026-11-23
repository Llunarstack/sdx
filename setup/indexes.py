#!/usr/bin/env python3
"""Build lookup indexes from scraped/trained data: artist aliases, RAG corpus, LoRA bank.

    python setup/indexes.py artist-index --data-root /workspace/data
    python setup/indexes.py rag-corpus --data-root /workspace/data
    python setup/indexes.py lora-subsets --manifest M --out /workspace/data/lora_subsets
    python setup/indexes.py lora-bank --bank-root /workspace/data/lora_bank --out /workspace/data/lora_bank/index.json

``artist-index``: every artist tag seen in booru downloads, so ``@AnyName`` in prompts resolves to the trained
spelling. ``rag-corpus``: point ``sample.py --local-rag-jsonl`` at it; SDX runs TF-IDF retrieval over the caption/tag
text and merges top facts into the prompt (see ``utils/prompt/rag.py``).
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

from scripts.scrape import ADAPTERS  # noqa: E402


def _manifests(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    """Explicit ``--manifest`` paths plus ``<data-root>/combined`` and per-site manifests, deduped."""
    found = [Path(m) for m in args.manifest]
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is not None:
        for sub in ("combined", *args.sites):
            p = data_root / sub / "manifest.jsonl"
            if p.is_file():
                found.append(p)
    return list(dict.fromkeys(found)), data_root


def _default_out(args: argparse.Namespace, data_root: Path | None, name: str) -> Path:
    if args.out:
        return Path(args.out)
    return (data_root or REPO_ROOT / "data") / name


def _artist_index(args: argparse.Namespace) -> int:
    from utils.prompt.text import build_from_manifests

    manifests, data_root = _manifests(args)
    if not manifests:
        print("No manifests found. Pass --data-root or --manifest.", file=sys.stderr)
        return 2
    reg = build_from_manifests(manifests)
    out = _default_out(args, data_root, "artist_index.json")
    reg.save(out)
    print(f"Indexed {len(reg):,} artists from {len(manifests)} manifest(s) -> {out}")
    return 0


def build_rag_corpus(manifest_paths: list[Path], out_path: Path, *, max_entries: int) -> int:
    seen: set[str] = set()
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for mp in manifest_paths:
            if not mp.is_file():
                continue
            with mp.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if written >= max_entries:
                        return written
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cap = (row.get("caption") or "").strip()
                    summary = (row.get("scene_summary") or "").strip()
                    key = (cap + "|" + summary).lower()
                    if (not cap and not summary) or key in seen:
                        continue
                    seen.add(key)
                    parts = [
                        f"{label}: {', '.join(row[field])}"
                        for label, field in (
                            ("characters", "character_tags"),
                            ("series", "copyright_tags"),
                            ("artist", "artist_tags"),
                        )
                        if row.get(field)
                    ]
                    if summary:
                        parts.append(summary)
                    if cap:
                        parts.append(f"tags: {cap}")
                    rec = {"text": ". ".join(parts), "caption": cap, "source": row.get("source", "")}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
    return written


def _rag_corpus(args: argparse.Namespace) -> int:
    manifests, data_root = _manifests(args)
    if not manifests:
        print("No manifests found.", file=sys.stderr)
        return 2
    out = _default_out(args, data_root, "rag_corpus.jsonl")
    n = build_rag_corpus(manifests, out, max_entries=int(args.max_entries))
    print(f"Wrote {n:,} RAG entries -> {out}")
    print(f"Use at inference: python sample.py --local-rag-jsonl {out} --prompt '...'")
    return 0


_STYLE_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("anime", ("anime", "manga", "cel shading", "toon shading", "1girl", "1boy", "solo")),
    ("digital_painting", ("digital painting", "digital art", "photoshop", "procreate", "illustration")),
    ("realistic", ("photorealistic", "photo realistic", "realistic", "photograph", "dslr")),
    ("3d_render", ("3d render", "blender", "octane", "unreal engine", "cgi", "ray tracing")),
    ("pixel_art", ("pixel art", "pixelated", "8-bit", "16-bit")),
    ("concept_art", ("concept art", "matte painting", "environment concept")),
]


def _style_buckets() -> list[tuple[str, tuple[str, ...]]]:
    from config.styles import MEDIUM_SPECS

    buckets = list(_STYLE_BUCKETS)
    known = {b[0] for b in buckets}
    for pack in MEDIUM_SPECS:
        kws = tuple(pack.keywords or ())[:6]
        if pack.id not in known and kws:
            buckets.append((pack.id, kws))
    return buckets


def _classify_style(row: dict, buckets: list[tuple[str, tuple[str, ...]]]) -> str | None:
    text = " ".join(str(row.get(k) or "") for k in ("caption", "tags", "booru_caption", "scene_summary")).lower()
    best_id, best_score = None, 0
    for style_id, keywords in buckets:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_id, best_score = style_id, score
    return best_id


def build_subsets(
    manifest: Path, out_dir: Path, *, min_artist_samples: int = 150, max_artists: int = 64, min_style_samples: int = 200
) -> dict:
    from utils.modeling.checkpoints import slugify_lora_key

    buckets = _style_buckets()
    artist_rows: dict[str, list[dict]] = defaultdict(list)
    style_rows: dict[str, list[dict]] = defaultdict(list)
    artist_counts: Counter[str] = Counter()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
        style = _classify_style(row, buckets)
        if style:
            style_rows[style].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"artists": {}, "styles": {}}

    def write(kind: str, key: str, subset: list[dict]) -> dict:
        path = out_dir / kind / key / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8")
        return {"manifest": str(path), "count": len(subset)}

    for slug, c in artist_counts.most_common(max_artists):
        if c >= min_artist_samples:
            meta["artists"][slug] = write("artist", slug, artist_rows[slug])
    for style_id, subset in style_rows.items():
        if len(subset) >= min_style_samples:
            meta["styles"][style_id] = write("style", style_id, subset)

    (out_dir / "subsets.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"LoRA subsets: {len(meta['artists'])} artists, {len(meta['styles'])} styles -> {out_dir}", flush=True)
    return meta


def _lora_subsets(args: argparse.Namespace) -> int:
    build_subsets(
        Path(args.manifest),
        Path(args.out),
        min_artist_samples=args.min_artist_samples,
        max_artists=args.max_artists,
        min_style_samples=args.min_style_samples,
    )
    return 0


_LORA_SUFFIXES = ("_lora.pt", "_lora.safetensors", ".safetensors", ".pt")


def _find_adapter_file(folder: Path) -> Path | None:
    for name in ("best_lora.pt", "latest_lora.pt", "adapter_lora.pt"):
        if (folder / name).is_file():
            return folder / name
    return next((p for p in sorted(folder.glob("*")) if p.is_file() and p.name.endswith(_LORA_SUFFIXES)), None)


def build_lora_bank(bank_root: Path, *, default_scale: float = 0.75):
    from utils.modeling.checkpoints import LoRABank, LoRAEntry, slugify_lora_key

    bank = LoRABank(root=bank_root)
    for kind, dest in (("artist", "artists"), ("style", "styles")):
        base = bank_root / kind
        if not base.is_dir():
            continue
        bucket = getattr(bank, dest)
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            adapter = _find_adapter_file(sub)
            if adapter is not None:
                bucket[slugify_lora_key(sub.name)] = LoRAEntry(
                    lora=adapter.relative_to(bank_root).as_posix(), default_scale=default_scale, role="style"
                )
    return bank


def _lora_bank(args: argparse.Namespace) -> int:
    root = Path(args.bank_root or REPO_ROOT / "data" / "lora_bank")
    out = Path(args.out or root / "index.json")
    bank = build_lora_bank(root, default_scale=args.default_scale)
    bank.save(out)
    print(f"LoRA bank index: {len(bank.artists)} artists, {len(bank.styles)} styles -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, helptext, out_help, extra in (
        ("artist-index", _artist_index, "Build artist alias index from scrape manifests", "artist_index.json", None),
        ("rag-corpus", _rag_corpus, "Build RAG corpus JSONL from scrape manifests", "rag_corpus.jsonl", 500_000),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--data-root", default=None, help="Scan combined + per-site manifests under this dir.")
        p.add_argument("--manifest", action="append", default=[], help="Explicit manifest.jsonl path (repeatable).")
        p.add_argument("--out", default=None, help=f"Output path (default: <data-root>/{out_help}).")
        if extra is not None:
            p.add_argument("--max-entries", type=int, default=extra)
        p.add_argument("--sites", nargs="*", default=sorted(ADAPTERS))
        p.set_defaults(fn=fn)

    p = sub.add_parser("lora-subsets", help="Build per-artist / per-style subset manifests for LoRA bank training")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default=None, help="Unused; kept for CLI symmetry.")
    p.add_argument("--out", required=True)
    p.add_argument("--min-artist-samples", type=int, default=150)
    p.add_argument("--max-artists", type=int, default=64)
    p.add_argument("--min-style-samples", type=int, default=200)
    p.set_defaults(fn=_lora_subsets)

    p = sub.add_parser("lora-bank", help="Scan trained LoRA adapters and build lora_bank/index.json")
    p.add_argument("--bank-root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--default-scale", type=float, default=0.75)
    p.set_defaults(fn=_lora_bank)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
