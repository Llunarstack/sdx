"""Clip retrieval: local library, scoring, optional Pexels API, web catalog metadata."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .types import ClipCandidate, RetrievalSource, ShotSpec
from .video_io import probe_video

__all__ = [
    "build_clip_candidate_from_path",
    "load_local_clip_library",
    "rank_clips_for_shot",
    "search_pexels_videos",
    "search_web_catalog",
]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2]


def build_clip_candidate_from_path(
    path: str | Path,
    *,
    source: RetrievalSource = RetrievalSource.LOCAL,
    title: str = "",
    tags: Optional[List[str]] = None,
    license: str = "user",
    url: str = "",
) -> ClipCandidate:
    p = Path(path)
    info = probe_video(p) if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".avi") else {}
    return ClipCandidate(
        source=source,
        path=str(p.resolve()),
        title=title or p.stem,
        tags=list(tags or []),
        duration_sec=float(info.get("duration_sec") or 0.0),
        fps=float(info.get("fps") or 24.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        license=license,
        url=url,
    )


def load_local_clip_library(
    root: str | Path,
    *,
    extensions: Sequence[str] = (".mp4", ".mov", ".webm", ".mkv"),
) -> List[ClipCandidate]:
    root = Path(root)
    if not root.is_dir():
        return []
    out: List[ClipCandidate] = []
    for ext in extensions:
        for p in root.rglob(f"*{ext}"):
            sidecar = p.with_suffix(p.suffix + ".json")
            meta: Dict[str, Any] = {}
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            out.append(
                build_clip_candidate_from_path(
                    p,
                    source=RetrievalSource.LOCAL,
                    title=str(meta.get("title") or p.stem),
                    tags=list(tags),
                    license=str(meta.get("license") or "local"),
                    url=str(meta.get("url") or ""),
                )
            )
    return out


def rank_clips_for_shot(
    shot: ShotSpec,
    candidates: Sequence[ClipCandidate],
    *,
    motion_scores: Optional[Dict[str, float]] = None,
) -> List[ClipCandidate]:
    """Score clips by tag/token overlap + duration fit + optional motion score."""
    motion_scores = motion_scores or {}
    q = set(_tokenize(shot.prompt)) | set(_tokenize(shot.motion_hint)) | set(_tokenize(shot.shot_type))
    ranked: List[ClipCandidate] = []
    for c in candidates:
        hay = set(_tokenize(c.title)) | set(_tokenize(" ".join(c.tags)))
        overlap = len(q & hay) / max(1, len(q))
        dur_fit = 1.0
        if c.duration_sec > 0 and shot.duration_sec > 0:
            ratio = min(c.duration_sec, shot.duration_sec) / max(c.duration_sec, shot.duration_sec)
            dur_fit = ratio
        motion = float(motion_scores.get(c.path, 0.5))
        want_motion = 0.6 if "static" not in shot.motion_hint.lower() else 0.3
        motion_fit = 1.0 - abs(motion - want_motion)
        score = 0.55 * overlap + 0.25 * dur_fit + 0.20 * motion_fit
        ranked.append(
            ClipCandidate(
                source=c.source,
                path=c.path,
                title=c.title,
                tags=list(c.tags),
                duration_sec=c.duration_sec,
                fps=c.fps,
                width=c.width,
                height=c.height,
                license=c.license,
                url=c.url,
                score=float(score),
                motion_score=motion,
            )
        )
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def search_pexels_videos(query: str, *, max_results: int = 5) -> List[ClipCandidate]:
    """
    Search Pexels video API when ``PEXELS_API_KEY`` is set.

    Returns metadata + download URL; does not download unless caller uses ``download_pexels_clip``.
    """
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if not key:
        return []
    try:
        import urllib.parse
        import urllib.request

        url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(
            {"query": query, "per_page": str(max(1, max_results))}
        )
        req = urllib.request.Request(url, headers={"Authorization": key})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: List[ClipCandidate] = []
    for vid in data.get("videos") or []:
        files = vid.get("video_files") or []
        best = None
        best_area = 0
        for f in files:
            w, h = int(f.get("width") or 0), int(f.get("height") or 0)
            if w * h > best_area and f.get("link"):
                best_area = w * h
                best = f
        if not best:
            continue
        out.append(
            ClipCandidate(
                source=RetrievalSource.PEXELS,
                path=str(best.get("link") or ""),
                title=str(vid.get("url") or query)[:120],
                tags=_tokenize(query),
                duration_sec=float(vid.get("duration") or 0.0),
                fps=24.0,
                width=int(best.get("width") or 0),
                height=int(best.get("height") or 0),
                license="pexels",
                url=str(vid.get("url") or best.get("link") or ""),
                score=0.5,
            )
        )
    return out


def download_remote_clip(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def search_web_catalog(
    query: str,
    *,
    catalog_path: str | Path = "",
    max_results: int = 5,
) -> List[ClipCandidate]:
    """
    Search a local JSON catalog of licensed web clips (metadata only).

    Catalog format: ``{"clips": [{"title", "tags", "path" or "url", "license"}]}``
    """
    cp = Path(catalog_path) if catalog_path else Path("data/video_catalog.json")
    if not cp.is_file():
        return []
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("clips") or data.get("videos") or []
    q = set(_tokenize(query))
    scored: List[ClipCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        hay = set(_tokenize(title)) | set(_tokenize(" ".join(str(t) for t in tags)))
        overlap = len(q & hay) / max(1, len(q))
        if overlap <= 0:
            continue
        path = str(row.get("path") or row.get("url") or "")
        scored.append(
            ClipCandidate(
                source=RetrievalSource.WEB_CATALOG,
                path=path,
                title=title,
                tags=[str(t) for t in tags],
                duration_sec=float(row.get("duration_sec") or 0.0),
                license=str(row.get("license") or "catalog"),
                url=str(row.get("url") or path),
                score=float(overlap),
            )
        )
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:max_results]
