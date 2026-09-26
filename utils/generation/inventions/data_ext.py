"""Data / board inventions 72–79."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ClipFaissIndexStub",
    "rating_filter_rows",
    "canonicalize_booru_tags",
    "mine_hard_negatives",
    "artist_upweight",
    "perceptual_hash",
    "dedup_paths_by_hash",
    "build_clip_faiss_plan",
]


@dataclass
class ClipFaissIndexStub:
    """Scaffold for CLIP+FAISS over board images (#72)."""

    dim: int = 768
    n_vectors: int = 0
    path: str = ""
    notes: list[str] = field(default_factory=list)

    def add(self, n: int = 1) -> None:
        self.n_vectors += int(n)

    def query_plan(self, text: str, top_k: int = 8) -> dict[str, Any]:
        return {
            "backend": "faiss_ip",
            "query": text,
            "top_k": top_k,
            "index_path": self.path,
            "status": "stub_until_embeddings_built",
        }


def build_clip_faiss_plan(*, index_path: str, model_id: str = "openai/clip-vit-large-patch14") -> dict[str, Any]:
    return {
        "steps": [
            "embed all corpus images with CLIP vision",
            "build FAISS IndexFlatIP or IVF",
            "at sample: embed prompt → top-k paths → InstantStyle",
        ],
        "model_id": model_id,
        "index_path": index_path,
    }


def rating_filter_rows(rows: list[dict[str, Any]], *, allow: set[str] | None = None) -> list[dict[str, Any]]:
    allow = allow or {"safe", "general", "sensitive", "questionable", "explicit"}
    out = []
    for r in rows:
        rating = str(r.get("rating") or r.get("rating_label") or "safe").lower()
        if rating in allow or rating[:1] in {a[:1] for a in allow}:
            out.append(r)
    return out


_TAG_ORDER = ("count", "subject", "character", "copyright", "artist", "style", "meta", "quality", "other")


def canonicalize_booru_tags(tags: str | list[str]) -> str:
    """Danbooru-ish tag order (#75)."""
    if isinstance(tags, str):
        parts = [t.strip() for t in re.split(r"[,\s]+", tags) if t.strip()]
    else:
        parts = [str(t).strip() for t in tags if str(t).strip()]
    buckets: dict[str, list[str]] = {k: [] for k in _TAG_ORDER}
    for t in parts:
        low = t.lower()
        if re.match(r"^\d+(girl|boy|girls|boys)$", low) or low in ("solo", "multiple_girls"):
            buckets["count"].append(t)
        elif low.startswith("by_") or low.endswith("_(style)") or "style" in low:
            buckets["artist"].append(t) if low.startswith("by_") else buckets["style"].append(t)
        elif low in ("masterpiece", "best_quality", "amazing_quality", "newest"):
            buckets["quality"].append(t)
        elif low in ("safe", "questionable", "explicit", "nsfw", "sfw"):
            buckets["meta"].append(t)
        elif low.startswith("character_") or low.endswith("_(character)"):
            buckets["character"].append(t)
        else:
            buckets["subject"].append(t)
    ordered: list[str] = []
    for k in _TAG_ORDER:
        ordered.extend(buckets[k])
    return ", ".join(dict.fromkeys(ordered))


def mine_hard_negatives(
    rows: list[dict[str, Any]],
    *,
    tag_key: str = "tags",
    anatomy_bad_key: str = "bad_anatomy",
) -> list[dict[str, Any]]:
    """Same tags, mark anatomy-bad as hard negatives (#76)."""
    pairs = []
    by_tags: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        tags = r.get(tag_key) or r.get("text") or ""
        key = canonicalize_booru_tags(tags if isinstance(tags, list) else str(tags))
        by_tags.setdefault(key, []).append(r)
    for key, group in by_tags.items():
        goods = [g for g in group if not g.get(anatomy_bad_key)]
        bads = [g for g in group if g.get(anatomy_bad_key)]
        for g in goods:
            for b in bads:
                pairs.append({"win": g, "lose": b, "tags": key})
    return pairs


def artist_upweight(artist_counts: Counter[str], artist: str, *, floor: float = 1.0, ceiling: float = 5.0) -> float:
    """Long-tail artist upweight (#78): rare artists get higher sample weight."""
    n = float(artist_counts.get(artist, 1))
    total = float(sum(artist_counts.values()) or 1)
    freq = n / total
    # inverse freq clipped
    w = 1.0 / max(freq, 1e-6) ** 0.5
    return float(max(floor, min(ceiling, w / 10.0)))


def perceptual_hash(path: str | Path, *, chunk: int = 8192) -> str:
    """Lightweight file hash proxy for dedup (#79); replace with pHash when available."""
    h = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        return ""
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def dedup_paths_by_hash(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        ph = perceptual_hash(path)
        if not ph or ph in seen:
            continue
        seen.add(ph)
        out.append(path)
    return out
