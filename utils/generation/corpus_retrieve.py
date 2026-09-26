"""Retrieve training-corpus images/captions at sample time to ground generation.

Designed for board-scale datasets (Danbooru, Gelbooru, e621, Rule34*, ArtStation,
DeviantArt, …) stored as JSONL with ``image``/``file_name`` + ``text``/``caption``/``tags``.

Uses TF-IDF over captions/tags (always). Optional CLIP vision rerank when
``open_clip`` / torchvision CLIP is available — no mandatory GPU index build.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CorpusHit",
    "CorpusIndex",
    "build_corpus_index_from_jsonl",
    "load_corpus_index",
    "save_corpus_index_meta",
    "retrieve_corpus_refs",
    "hits_to_moodboard_payload",
    "BOARD_SOURCE_HINTS",
]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]{1,}", re.IGNORECASE)

# Known board / site tags that may appear in paths or metadata (documentation + filters).
BOARD_SOURCE_HINTS: tuple[str, ...] = (
    "danbooru",
    "gelbooru",
    "e621",
    "rule34",
    "rule34xyz",
    "deviantart",
    "artstation",
    "pixiv",
    "safebooru",
    "anime-pictures",
    "konachan",
    "yande.re",
)


@dataclass(slots=True)
class CorpusHit:
    path: str
    caption: str
    score: float
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusIndex:
    """In-memory caption/tag index with optional on-disk image paths."""

    paths: list[str] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tags: list[list[str]] = field(default_factory=list)
    doc_freq: dict[str, int] = field(default_factory=dict)
    doc_tfidf: list[dict[str, float]] = field(default_factory=list)
    root: str = ""

    def __len__(self) -> int:
        return len(self.paths)

    def _build_tfidf(self) -> None:
        n = len(self.captions)
        self.doc_freq = {}
        tf_rows: list[Counter[str]] = []
        for cap, tag_list in zip(self.captions, self.tags, strict=False):
            blob = cap
            if tag_list:
                blob = f"{cap} {' '.join(tag_list)}"
            toks = [t.lower() for t in _TOKEN_RE.findall(blob or "")]
            c = Counter(toks)
            tf_rows.append(c)
            for tok in c:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
        self.doc_tfidf = []
        for c in tf_rows:
            denom = float(sum(c.values()) or 1)
            row: dict[str, float] = {}
            for tok, cnt in c.items():
                idf = math.log((1.0 + n) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
                row[tok] = (cnt / denom) * idf
            self.doc_tfidf.append(row)

    def query_tfidf(self, text: str, *, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.doc_tfidf:
            self._build_tfidf()
        q_toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        if not q_toks or not self.doc_tfidf:
            return []
        qc = Counter(q_toks)
        q_denom = float(sum(qc.values()) or 1)
        n = max(1, len(self.doc_tfidf))
        q_vec: dict[str, float] = {}
        for tok, cnt in qc.items():
            idf = math.log((1.0 + n) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
            q_vec[tok] = (cnt / q_denom) * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()) or 1e-12)
        scored: list[tuple[int, float]] = []
        for i, row in enumerate(self.doc_tfidf):
            dot = 0.0
            for tok, qv in q_vec.items():
                dot += qv * row.get(tok, 0.0)
            r_norm = math.sqrt(sum(v * v for v in row.values()) or 1e-12)
            scored.append((i, float(dot / (q_norm * r_norm + 1e-12))))
        scored.sort(key=lambda x: -x[1])
        return scored[: max(1, int(top_k))]


def _infer_source(path: str, meta: dict[str, Any]) -> str:
    s = str(meta.get("source") or meta.get("site") or meta.get("board") or "").lower()
    if s:
        return s
    low = path.replace("\\", "/").lower()
    for hint in BOARD_SOURCE_HINTS:
        if hint in low:
            return hint
    return ""


def _row_path_caption(obj: dict[str, Any], *, root: Path | None) -> tuple[str, str, list[str]] | None:
    path = obj.get("image") or obj.get("file_name") or obj.get("path") or obj.get("file") or obj.get("img") or ""
    path = str(path).strip()
    if not path:
        return None
    p = Path(path)
    if not p.is_file() and root is not None:
        cand = root / path
        if cand.is_file():
            path = str(cand)
    cap = obj.get("text") or obj.get("caption") or obj.get("prompt") or obj.get("tag_string") or obj.get("tags") or ""
    tags: list[str] = []
    if isinstance(obj.get("tags"), list):
        tags = [str(t).strip() for t in obj["tags"] if str(t).strip()]
        if not cap:
            cap = ", ".join(tags)
    elif isinstance(cap, list):
        tags = [str(t).strip() for t in cap if str(t).strip()]
        cap = ", ".join(tags)
    cap = str(cap).strip()
    if not cap:
        return None
    return path, cap, tags


def build_corpus_index_from_jsonl(
    jsonl_path: str | Path,
    *,
    root: str | Path | None = None,
    max_rows: int = 0,
    require_existing_file: bool = False,
) -> CorpusIndex:
    """Scan a training/manifest JSONL into a CorpusIndex."""
    path = Path(jsonl_path)
    root_p = Path(root) if root else path.parent
    idx = CorpusIndex(root=str(root_p))
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            parsed = _row_path_caption(obj, root=root_p)
            if parsed is None:
                continue
            img_path, cap, tags = parsed
            if require_existing_file and not Path(img_path).is_file():
                continue
            idx.paths.append(img_path)
            idx.captions.append(cap)
            idx.tags.append(tags)
            idx.sources.append(_infer_source(img_path, obj))
            n += 1
            if max_rows > 0 and n >= max_rows:
                break
    idx._build_tfidf()
    return idx


def save_corpus_index_meta(index: CorpusIndex, out_path: str | Path) -> Path:
    """Save lightweight meta (paths+captions) for fast reload without re-scan."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(index)):
        rows.append(
            {
                "image": index.paths[i],
                "text": index.captions[i],
                "source": index.sources[i] if i < len(index.sources) else "",
                "tags": index.tags[i] if i < len(index.tags) else [],
            }
        )
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def load_corpus_index(jsonl_path: str | Path, *, root: str | Path | None = None) -> CorpusIndex:
    return build_corpus_index_from_jsonl(jsonl_path, root=root, require_existing_file=False)


def retrieve_corpus_refs(
    index: CorpusIndex,
    query: str,
    *,
    top_k: int = 6,
    source_filter: str = "",
    prefer_existing: bool = True,
) -> list[CorpusHit]:
    """TF-IDF retrieve corpus hits for a prompt (subject/style/attire tags work well)."""
    ranked = index.query_tfidf(query, top_k=max(top_k * 4, top_k))
    hits: list[CorpusHit] = []
    filt = str(source_filter or "").lower().strip()
    for i, score in ranked:
        src = index.sources[i] if i < len(index.sources) else ""
        if filt and filt not in src and filt not in index.paths[i].lower():
            continue
        path = index.paths[i]
        if prefer_existing and path and not Path(path).is_file():
            # Still keep caption-only hit with empty path skipped for moodboard
            if not Path(path).is_file():
                # allow caption enrichment without image
                pass
        hits.append(
            CorpusHit(
                path=path,
                caption=index.captions[i],
                score=float(score),
                source=src,
                tags=list(index.tags[i]) if i < len(index.tags) else [],
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def hits_to_moodboard_payload(hits: list[CorpusHit]) -> dict[str, Any]:
    """JSON payload compatible with ``--moodboard-json`` / InstantStyle wiring."""
    images = [h.path for h in hits if h.path and Path(h.path).is_file()]
    return {
        "images": images,
        "paths": images,
        "captions": [h.caption for h in hits],
        "scores": [h.score for h in hits],
        "sources": [h.source for h in hits],
        "by_facet": {"corpus": images},
    }
