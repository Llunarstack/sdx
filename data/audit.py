"""Streaming audit of the dataset ``train.py`` reads (folder of image + caption files, or a JSONL manifest).

Uses the training loader itself (``training.loop.build_train_dataset``), so it sees exactly the rows, path
resolution and caption processing a run sees. Per image it keeps only hashes and a few counters, so it runs over
datasets that do not fit in RAM; ``limit`` audits a seeded random sample.

``python -m scripts.tools readiness dataset --data-path ...`` is the CLI.
"""

from __future__ import annotations

import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .manifest_utils import iter_manifest_rows

_SCAN_EXTS = (".png", ".jpg", ".jpeg", ".webp")  # Text2ImageDataset folder scan
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}
_LOSSY_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}  # convert("RGB") clips these
_EXAMPLES = 5
# Audited rows the loader would replace; with the rows the scan drops they make up the bad fraction.
BAD_KEYS = ("missing_file", "corrupt", "loader_failed")


def _dataset_source(cfg) -> Path:
    src = getattr(cfg, "manifest_jsonl", None) or getattr(cfg, "data_path", None)
    if not src:
        raise ValueError("Set --data-path or --manifest-jsonl")
    return Path(src)


def _scan_dirs(root: Path) -> list[Path]:
    return [root] + sorted(d for d in root.iterdir() if d.is_dir())


def dataset_fingerprint(cfg) -> str:
    """Short form of ``training.loop.dataset_fingerprint`` (what ``run_manifest.json`` records) for the run's data."""
    from training.loop import dataset_fingerprint as _fingerprint

    return _fingerprint(_dataset_source(cfg))["sha256"][:16]


def _scan_drops(src: Path, examples: dict[str, list]) -> Counter:
    """Rows the dataset scan drops without a word (minor-guard drops are counted by the dataset itself)."""
    c: Counter = Counter()

    def note(key: str, what: str) -> None:
        c[key] += 1
        if len(examples[key]) < _EXAMPLES:
            examples[key].append(what)

    if src.suffix.lower() == ".jsonl":
        for i, d in enumerate(iter_manifest_rows(src)):
            if not isinstance(d, dict):
                note("missing_image_field", f"line {i + 1}: not an object")
            elif not (d.get("image_path") or d.get("path") or d.get("image")):
                note("missing_image_field", f"line {i + 1}")
            elif not (d.get("caption") or d.get("text") or d.get("instruction")):
                note("missing_caption", f"line {i + 1}")
        return c
    for d in _scan_dirs(src):
        for p in d.glob("*"):
            ext = p.suffix.lower()
            if ext in (".gif", ".bmp"):
                note("ignored_extension", str(p))
            if ext not in _SCAN_EXTS:
                continue
            cap = p.with_suffix(".txt")
            if not cap.exists():
                cap = p.with_name(p.stem + ".caption")
            if not cap.exists():
                note("missing_caption", str(p))
            elif not cap.read_text(encoding="utf-8", errors="ignore").strip().split("\n")[0].strip():
                note("empty_caption", str(p))
    return c


def _tokenizer(name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # not cached locally: fall back to a word/punctuation count
        return None


def _token_len(tok, caption: str) -> int:
    if tok is None:
        return len(re.findall(r"\w+|[^\w\s]", caption)) + 1
    return len(tok(caption, truncation=False)["input_ids"])


def _dhash(img: Image.Image) -> int:
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    px = g.tobytes()
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def _near_dup_groups(hashes: dict[int, int], max_bits: int) -> list[list[int]]:
    """Groups of indices whose 64-bit dHashes differ in <= ``max_bits`` bits (4x16-bit LSH bands, exact for <= 3)."""
    parent = {i: i for i in hashes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(4):
        buckets: defaultdict[int, list[int]] = defaultdict(list)
        for i, h in hashes.items():
            buckets[(h >> (16 * band)) & 0xFFFF].append(i)
        for members in buckets.values():
            members = members[:2000]  # ponytail: degenerate buckets (flat images) capped; O(k^2) inside a bucket
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    if bin(hashes[i] ^ hashes[j]).count("1") <= max_bits:
                        parent[find(i)] = find(j)
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in hashes:
        groups[find(i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def _val_indices(n: int, cfg) -> set[int]:
    """The val rows of ``train.py``'s ``random_split`` (same lengths and seed)."""
    import torch

    val_split = float(getattr(cfg, "val_split", 0.0) or 0.0)
    if not 0 < val_split < 1 or n < 2:
        return set()
    n_val = max(1, int(n * val_split))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(cfg.global_seed))).tolist()
    return set(perm[n - n_val :])


def audit_dataset(
    cfg,
    *,
    limit: int = 0,
    seed: int = 0,
    max_bad_frac: float = 0.01,
    min_caption_tokens: int = 3,
    max_crop_loss: float = 0.4,
    near_dup_bits: int = 3,
    check_loader: bool = True,
    progress=None,
) -> dict[str, Any]:
    """Audit report (JSON-able). ``report["ok"]`` is False when the bad-row fraction exceeds ``max_bad_frac``."""
    from training.loop import build_train_dataset

    from data.caption_utils import merge_region_captions_into_caption

    src = _dataset_source(cfg)
    data_root = getattr(cfg, "data_path", None) if getattr(cfg, "manifest_jsonl", None) else None
    buckets = [tuple(b) for b in (getattr(cfg, "resolution_buckets", None) or [])]
    ds = build_train_dataset(cfg, str(src), data_root=data_root, latent_dir=None, res_buckets=buckets)
    examples: defaultdict[str, list] = defaultdict(list)
    counts = _scan_drops(src, examples)
    counts["minor_guard_dropped"] = int(getattr(ds, "n_blocked", 0))
    n = len(ds)
    idxs = list(range(n)) if not limit or limit >= n else sorted(random.Random(seed).sample(range(n), int(limit)))

    tok = _tokenizer(str(getattr(cfg, "text_encoder", "")))
    lengths = getattr(cfg, "curriculum_max_lengths", None) or [300]
    token_limit = int(max(lengths))
    clip_limit = 77 if str(getattr(cfg, "text_encoder_mode", "t5")) != "t5" else 0
    modes: Counter = Counter()
    sha_groups: defaultdict[str, list[int]] = defaultdict(list)
    dhashes: dict[int, int] = {}
    tok_lens: list[int] = []

    bad_idx: set[int] = set()

    def note(key: str, i: int, extra: str = "") -> None:
        counts[key] += 1
        if key in BAD_KEYS:
            bad_idx.add(i)
        if len(examples[key]) < _EXAMPLES:
            examples[key].append(ds.samples[i]["path"] + (f" ({extra})" if extra else ""))

    for k, i in enumerate(idxs):
        if progress and k % 500 == 0:
            progress(k, len(idxs))
        s = ds.samples[i]
        caption = s["caption"]
        if check_loader:
            try:
                caption = ds._getitem_impl(i)["caption"]  # the processed caption the text encoder sees
            except Exception as e:
                note("loader_failed", i, type(e).__name__)
        else:
            caption = ds._process_caption(merge_region_captions_into_caption(caption, s.get("regions")))[0]
        path = ds._resolve_media_path(s["path"])
        if path is None or not path.is_file():
            note("missing_file", i)
        else:
            try:
                data = path.read_bytes()
                sha_groups[hashlib.sha1(data).hexdigest()].append(i)
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
                    mode, (w, h) = img.mode, img.size
                    modes[mode] += 1
                    if mode in _ALPHA_MODES or (mode == "P" and "transparency" in img.info):
                        note("alpha_flattened", i, mode)
                    if mode in _LOSSY_MODES:
                        note("lossy_mode_conversion", i, mode)
                    dhashes[i] = _dhash(img)
                th, tw = buckets[ds._bucket_assign[i]] if buckets else (ds.image_size, ds.image_size)
                if w < tw or h < th:
                    note("upscaled_small_image", i, f"{w}x{h} < {tw}x{th}")
                ratio = (w / max(h, 1)) / (tw / th)
                if 1 - min(ratio, 1 / ratio) > max_crop_loss:
                    note("extreme_aspect_crop", i, f"{w}x{h} -> {tw}x{th}, loses {1 - min(ratio, 1 / ratio):.0%}")
            except Exception as e:
                note("corrupt", i, type(e).__name__)
        n_tok = _token_len(tok, caption)
        tok_lens.append(n_tok)
        if not caption.strip():
            note("empty_after_processing", i)
        elif n_tok < min_caption_tokens:
            note("short_caption", i, repr(caption[:60]))
        if n_tok > token_limit:
            note("caption_truncated", i, f"{n_tok} > {token_limit} tokens")
        if clip_limit and n_tok > clip_limit:
            counts["caption_over_clip_77"] += 1

    exact = [g for g in sha_groups.values() if len(g) > 1]
    sha_of = {i: h for h, g in sha_groups.items() for i in g}
    near = [g for g in _near_dup_groups(dhashes, near_dup_bits) if len({sha_of.get(i) for i in g}) > 1]
    counts["exact_duplicate_rows"] = sum(len(g) - 1 for g in exact)
    counts["near_duplicate_rows"] = sum(len(g) - 1 for g in near)
    examples["exact_duplicates"] = [[ds.samples[i]["path"] for i in g[:4]] for g in exact[:_EXAMPLES]]
    examples["near_duplicates"] = [[ds.samples[i]["path"] for i in g[:4]] for g in near[:_EXAMPLES]]
    val = _val_indices(n, cfg)
    if val:
        leaks = [g for g in exact + near if any(i in val for i in g) and any(i not in val for i in g)]
        counts["train_val_overlap_groups"] = len(leaks)
        examples["train_val_overlap"] = [[ds.samples[i]["path"] for i in g[:4]] for g in leaks[:_EXAMPLES]]

    raw_total = n + sum(counts[k] for k in ("missing_caption", "empty_caption", "missing_image_field"))
    raw_total += counts["minor_guard_dropped"]
    scan_bad = sum(counts[k] for k in ("missing_caption", "empty_caption", "missing_image_field"))
    bad_frac = (scan_bad + len(bad_idx) * n / max(1, len(idxs))) / max(1, raw_total)
    tok_lens.sort()
    return {
        "ok": bad_frac <= max_bad_frac and n > 0,
        "source": str(src),
        "fingerprint": dataset_fingerprint(cfg),
        "rows_total": raw_total,
        "rows_loadable_by_scan": n,
        "rows_audited": len(idxs),
        "sampled": len(idxs) < n,
        "bad_frac": round(bad_frac, 6),
        "max_bad_frac": max_bad_frac,
        "image_modes": dict(modes),
        "loader_converts_to_rgb": True,
        "token_limit": token_limit,
        "tokenizer": "exact" if tok is not None else "approx (tokenizer not cached locally)",
        "caption_tokens_p50_p95_max": [tok_lens[len(tok_lens) // 2], tok_lens[int(len(tok_lens) * 0.95)], tok_lens[-1]]
        if tok_lens
        else [],
        "counts": {k: v for k, v in sorted(counts.items()) if v},
        "examples": {k: v for k, v in sorted(examples.items()) if v},
    }


def format_audit(r: dict[str, Any]) -> str:
    lines = [
        f"Dataset audit: {r['source']}  [{'PASS' if r['ok'] else 'FAIL'}]",
        f"  rows: {r['rows_total']} total, {r['rows_loadable_by_scan']} kept by the loader scan, "
        f"{r['rows_audited']} audited{' (sample)' if r['sampled'] else ''}",
        f"  bad fraction: {r['bad_frac']:.4f} (max {r['max_bad_frac']})   modes: {r['image_modes']} (loader -> RGB)",
        f"  caption tokens p50/p95/max: {r['caption_tokens_p50_p95_max']} (limit {r['token_limit']}, {r['tokenizer']})",
    ]
    lines += [f"  {k}: {v}" for k, v in r["counts"].items()]
    for k, v in r["examples"].items():
        lines.append(f"  e.g. {k}: {v[:3]}")
    return "\n".join(lines)
