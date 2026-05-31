"""Programmatic data curation (dedup, filters) for training manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


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
    bad_words: Tuple[str, ...] = ()
    min_weight: float = 0.0
    min_clip_sim: float = 0.0
    min_aesthetic_proxy: float = 0.0
    image_root: Optional[Path] = None


def _perceptual_hash(path: Path, size: int = 8) -> str:
    try:
        import imagehash
        from PIL import Image

        return str(imagehash.phash(Image.open(path), hash_size=size))
    except Exception:
        return ""


def _file_md5(path: Path) -> str:
    try:
        from sdx_native.native_tools import file_md5_hex

        return file_md5_hex(path, prefer_native_md5=True)
    except Exception:
        pass
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _resolve_image(path_str: str, *, jsonl_dir: Path, image_root: Optional[Path]) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if image_root is not None:
        return (image_root / p).resolve()
    return (jsonl_dir / p).resolve()


def filter_jsonl_row(
    row: Dict[str, Any],
    cfg: FilterConfig,
    *,
    seen_hashes: Set[str],
    jsonl_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (row, reason) or (None, drop_reason)."""
    path = row.get("image_path") or row.get("path") or row.get("image") or ""
    cap = (row.get("caption") or row.get("text") or "").strip()
    if not path or not cap:
        return None, "missing_path_or_caption"
    if cfg.dedup:
        img_path = _resolve_image(str(path), jsonl_dir=jsonl_dir, image_root=cfg.image_root)
        h = _perceptual_hash(img_path) if cfg.dedup == "phash" else _file_md5(img_path)
        if not h and cfg.dedup == "phash":
            h = _file_md5(img_path)
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


def filter_jsonl_file(
    input_path: Union[str, Path],
    *,
    config: Optional[FilterConfig] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Dict[str, Any]], FilterStats]:
    """Filter a JSONL manifest; optionally write output."""
    cfg = config or FilterConfig()
    inp = Path(input_path)
    jsonl_dir = inp.parent
    seen: Set[str] = set()
    stats = FilterStats()
    out_rows: List[Dict[str, Any]] = []
    with inp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            kept, reason = filter_jsonl_row(row, cfg, seen_hashes=seen, jsonl_dir=jsonl_dir)
            if kept is None:
                if reason == "dup":
                    stats.dropped_dup += 1
                elif reason == "caption_len":
                    stats.dropped_caption += 1
                elif reason == "bad_word":
                    stats.dropped_bad_words += 1
                elif reason == "weight":
                    stats.dropped_weight += 1
                elif reason == "clip_sim":
                    stats.dropped_clip += 1
                elif reason == "aesthetic":
                    stats.dropped_aesthetic += 1
                continue
            out_rows.append(kept)
            stats.kept += 1
    if output_path:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_rows, stats


__all__ = ["FilterConfig", "FilterStats", "filter_jsonl_file", "filter_jsonl_row"]
