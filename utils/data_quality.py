"""Manifest curation: perceptual/MD5 dedup plus caption, weight, CLIP and aesthetic filters."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FilterStats:
    kept: int = 0
    dropped_dup: int = 0
    dropped_caption: int = 0
    dropped_bad_words: int = 0
    dropped_weight: int = 0
    dropped_clip: int = 0
    dropped_aesthetic: int = 0

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_dup
            + self.dropped_caption
            + self.dropped_bad_words
            + self.dropped_weight
            + self.dropped_clip
            + self.dropped_aesthetic
        )


@dataclass(slots=True)
class FilterConfig:
    dedup: str = ""  # "", "phash", "md5"
    min_caption_len: int = 0
    max_caption_len: int = 0
    bad_words: tuple[str, ...] = ()
    min_weight: float = 0.0
    min_clip_sim: float = 0.0
    min_aesthetic_proxy: float = 0.0
    image_root: Path | None = None
    native_md5: bool = True


def perceptual_hash(path: Path, size: int = 8) -> str:
    """pHash hex string, or ``""`` when imagehash/PIL cannot read the file."""
    try:
        import imagehash
        from PIL import Image

        return str(imagehash.phash(Image.open(path), hash_size=size))
    except Exception:
        return ""


def file_md5(path: Path, *, prefer_native: bool = True) -> str:
    """Streaming MD5 hex (Rust ``file-md5`` when built); ``""`` when the file cannot be read."""
    from utils.native import file_md5_hex

    return file_md5_hex(path, prefer_native_md5=prefer_native)


def _resolve_image(path_str: str, *, jsonl_dir: Path, image_root: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if image_root is not None:
        return (image_root / p).resolve()
    return (jsonl_dir / p).resolve()


def filter_jsonl_row(
    row: dict[str, Any],
    cfg: FilterConfig,
    *,
    seen_hashes: set[str],
    jsonl_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Return (row, reason) or (None, drop_reason)."""
    path = row.get("image_path") or row.get("path") or row.get("image") or ""
    cap = (row.get("caption") or row.get("text") or "").strip()
    if not path or not cap:
        return None, "missing_path_or_caption"
    if cfg.dedup:
        img_path = _resolve_image(str(path), jsonl_dir=jsonl_dir, image_root=cfg.image_root)
        h = perceptual_hash(img_path) if cfg.dedup == "phash" else ""
        h = h or file_md5(img_path, prefer_native=cfg.native_md5)
        if h and h in seen_hashes:
            return None, "dup"
        if h:
            seen_hashes.add(h)
    if cfg.min_caption_len and len(cap) < cfg.min_caption_len:
        return None, "caption_len"
    if cfg.max_caption_len and len(cap) > cfg.max_caption_len:
        return None, "caption_len"
    if cfg.bad_words and any(w in cap.lower() for w in cfg.bad_words):
        return None, "bad_word"
    w = float(row.get("weight", row.get("aesthetic_score", 1.0)))
    if cfg.min_weight > 0 and w < cfg.min_weight:
        return None, "weight"
    if cfg.min_clip_sim > 0 and "clip_sim" in row:
        try:
            if float(row["clip_sim"]) < cfg.min_clip_sim:
                return None, "clip_sim"
        except (TypeError, ValueError):
            pass
    if cfg.min_aesthetic_proxy > 0 and "aesthetic_proxy" in row:
        try:
            if float(row["aesthetic_proxy"]) < cfg.min_aesthetic_proxy:
                return None, "aesthetic"
        except (TypeError, ValueError):
            pass
    return row, ""


_DROP_COUNTERS = {
    "dup": "dropped_dup",
    "caption_len": "dropped_caption",
    "bad_word": "dropped_bad_words",
    "weight": "dropped_weight",
    "clip_sim": "dropped_clip",
    "aesthetic": "dropped_aesthetic",
}


def filter_rows(
    rows: Iterable[dict[str, Any]], cfg: FilterConfig, *, base_dir: Path
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Apply :func:`filter_jsonl_row` to ``rows`` (relative image paths resolve against ``base_dir``)."""
    seen: set[str] = set()
    stats = FilterStats()
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        kept, reason = filter_jsonl_row(row, cfg, seen_hashes=seen, jsonl_dir=base_dir)
        if kept is None:
            counter = _DROP_COUNTERS.get(reason)
            if counter:
                setattr(stats, counter, getattr(stats, counter) + 1)
            continue
        out_rows.append(kept)
        stats.kept += 1
    return out_rows, stats


def filter_jsonl_file(
    input_path: str | Path,
    *,
    config: FilterConfig | None = None,
    output_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Filter a JSONL manifest; optionally write output."""
    from data.manifest_utils import iter_manifest_rows

    inp = Path(input_path)
    out_rows, stats = filter_rows(iter_manifest_rows(inp), config or FilterConfig(), base_dir=inp.parent)
    if output_path:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_rows, stats


__all__ = [
    "FilterConfig",
    "FilterStats",
    "file_md5",
    "filter_jsonl_file",
    "filter_jsonl_row",
    "filter_rows",
    "perceptual_hash",
]
