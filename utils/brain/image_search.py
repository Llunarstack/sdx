"""
Online **image search** for Visual Brain (RAG-style reference gathering).

Uses DuckDuckGo image search via stdlib ``urllib`` (no API key). Falls back to
Wikimedia Commons when DDG is blocked. Gracefully returns empty on network failure.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(slots=True)
class ImageSearchHit:
    url: str
    title: str = ""
    source: str = "duckduckgo"
    local_path: str = ""


@dataclass(slots=True)
class ImageSearchResult:
    query: str
    hits: List[ImageSearchHit] = field(default_factory=list)
    notes: str = ""


_UA = "Mozilla/5.0 (compatible; SDX-VisualBrain/1.0)"


def _http_get(url: str, *, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ddg_vqd(query: str) -> Optional[str]:
    q = urllib.parse.quote_plus(query)
    html = _http_get(f"https://duckduckgo.com/?q={q}&iax=images&ia=images").decode("utf-8", errors="ignore")
    for pat in (r"vqd=([\d-]+)", r'"vqd"\s*:\s*"([^"]+)"', r"vqd='([^']+)'"):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


def search_images_duckduckgo(query: str, *, max_results: int = 5) -> ImageSearchResult:
    """Return image URLs from DuckDuckGo image search."""
    out = ImageSearchResult(query=str(query).strip())
    if not out.query:
        out.notes = "empty query"
        return out
    try:
        vqd = _ddg_vqd(out.query)
        if not vqd:
            out.notes = "no vqd token"
            return out
        q = urllib.parse.quote_plus(out.query)
        v = urllib.parse.quote_plus(vqd)
        js_url = f"https://duckduckgo.com/i.js?q={q}&vqd={v}&o=json"
        raw = _http_get(js_url).decode("utf-8", errors="ignore")
        data = json.loads(raw)
        for row in (data.get("results") or [])[: max(1, int(max_results))]:
            url = str(row.get("image") or row.get("thumbnail") or "").strip()
            if url:
                out.hits.append(
                    ImageSearchHit(
                        url=url,
                        title=str(row.get("title") or "")[:200],
                        source="duckduckgo",
                    )
                )
        if not out.hits:
            out.notes = "no results"
    except Exception as exc:
        out.notes = f"ddg error: {exc}"
    return out


def search_images_wikimedia(query: str, *, max_results: int = 5) -> ImageSearchResult:
    """Fallback: Wikimedia Commons file search (CC images)."""
    out = ImageSearchResult(query=str(query).strip())
    if not out.query:
        return out
    try:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {out.query}",
                "gsrlimit": str(max(1, int(max_results))),
                "prop": "imageinfo",
                "iiprop": "url",
            }
        )
        raw = _http_get(f"https://commons.wikimedia.org/w/api.php?{params}")
        data = json.loads(raw.decode("utf-8"))
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if infos and infos[0].get("url"):
                out.hits.append(
                    ImageSearchHit(
                        url=str(infos[0]["url"]),
                        title=str(page.get("title") or "")[:200],
                        source="wikimedia",
                    )
                )
    except Exception as exc:
        out.notes = f"wikimedia error: {exc}"
    return out


def search_reference_images(
    query: str,
    *,
    max_results: int = 5,
    allow_web: bool = True,
) -> ImageSearchResult:
    """Search online; try DDG then Wikimedia."""
    if not allow_web or not str(query or "").strip():
        return ImageSearchResult(query=str(query or ""), notes="web disabled or empty")
    res = search_images_duckduckgo(query, max_results=max_results)
    if res.hits:
        return res
    wiki = search_images_wikimedia(query, max_results=max_results)
    if wiki.hits:
        return wiki
    if not res.notes:
        res.notes = wiki.notes or "no hits"
    return res


def download_search_hits(
    hits: List[ImageSearchHit],
    output_dir: Path,
    *,
    max_download: int = 5,
) -> List[ImageSearchHit]:
    """Download remote URLs to *output_dir*; set ``local_path`` on success."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: List[ImageSearchHit] = []
    for i, hit in enumerate(hits[: max(1, int(max_download))]):
        if not hit.url:
            continue
        ext = ".jpg"
        if ".png" in hit.url.lower():
            ext = ".png"
        dest = out_dir / f"search_{i:02d}{ext}"
        try:
            data = _http_get(hit.url, timeout=25.0)
            if len(data) < 512:
                continue
            dest.write_bytes(data)
            updated.append(
                ImageSearchHit(
                    url=hit.url,
                    title=hit.title,
                    source=hit.source,
                    local_path=str(dest),
                )
            )
        except Exception:
            continue
    return updated


__all__ = [
    "ImageSearchHit",
    "ImageSearchResult",
    "download_search_hits",
    "search_images_duckduckgo",
    "search_images_wikimedia",
    "search_reference_images",
]
