"""Reverse image search via web upload (SauceNAO + TinEye) — no API keys required.

Optional API keys in secret.txt speed up SauceNAO/TinEye when present.
Booru site credentials (danbooru/e621) from secret.txt are used to fetch tags after a match.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from utils.caption.api_keys import get_api_key

_BOORU_URL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"danbooru\.donmai\.us/posts/(\d+)", re.I), "danbooru"),
    (re.compile(r"safebooru\.donmai\.us/posts/(\d+)", re.I), "danbooru"),
    (re.compile(r"e621\.net/posts/(\d+)", re.I), "e621"),
    (
        re.compile(
            r"(?:pixiv\.net/(?:en/)?artworks/(\d+)|pixiv\.net/member_illust\.php\?[^#]*illust_id=(\d+))",
            re.I,
        ),
        "pixiv",
    ),
]

_RATE_LOCK = threading.Lock()
_LAST_REQ: dict[str, float] = {"saucenao": 0.0, "tineye": 0.0}
_MIN_INTERVAL = {"saucenao": 3.5, "tineye": 2.0}


@dataclass
class ReverseHit:
    similarity: float
    site: str
    site_id: str
    engine: str = "saucenao"
    author: str = ""
    material: str = ""  # copyright / series hint
    characters: List[str] = field(default_factory=list)
    external_url: str = ""
    raw: dict = field(default_factory=dict)


def parse_source_url(url: str) -> tuple[str, str]:
    """Extract ``(site, post_id)`` from a booru/pixiv URL."""
    u = (url or "").strip()
    if not u:
        return "", ""
    for pat, site in _BOORU_URL_PATTERNS:
        m = pat.search(u)
        if not m:
            continue
        post_id = next((g for g in m.groups() if g), "")
        if post_id:
            return site, post_id
    return "", ""


def _rate_limit(engine: str) -> None:
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL.get(engine, 2.0) - (now - _LAST_REQ.get(engine, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST_REQ[engine] = time.monotonic()


def _iter_urls_from_tineye_match(match: dict) -> Iterable[str]:
    if match.get("image_url"):
        yield str(match["image_url"])
    for backlink in match.get("backlinks") or []:
        if backlink.get("url"):
            yield str(backlink["url"])
        if backlink.get("backlink"):
            yield str(backlink["backlink"])


def _hits_from_saucenao_json(payload: dict) -> List[ReverseHit]:
    hits: List[ReverseHit] = []
    for result in payload.get("results") or []:
        header = result.get("header") or {}
        data = result.get("data") or {}
        sim = float(header.get("similarity") or 0.0)
        ext = data.get("ext_urls") or []
        ext_url = str(ext[0]) if ext else ""
        site = ""
        site_id = ""
        author = str(data.get("member_name") or data.get("author_name") or "")
        material = str(data.get("material") or data.get("source") or "")
        chars: List[str] = []
        if "danbooru_id" in data:
            site = "danbooru"
            site_id = str(data["danbooru_id"])
        elif "pixiv_id" in data:
            site = "pixiv"
            site_id = str(data["pixiv_id"])
        elif ext_url:
            site, site_id = parse_source_url(ext_url)
            if not site and "e621" in ext_url:
                site = "e621"
            elif not site and "danbooru" in ext_url:
                site = "danbooru"
        char = data.get("char") or data.get("characters")
        if isinstance(char, str) and char.strip():
            chars = [c.strip() for c in char.split(",") if c.strip()]
        hits.append(
            ReverseHit(
                similarity=sim,
                site=site,
                site_id=site_id,
                engine="saucenao",
                author=author,
                material=material,
                characters=chars,
                external_url=ext_url,
                raw=data,
            )
        )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def _hits_from_saucenao_html(html: str) -> List[ReverseHit]:
    """Parse SauceNAO HTML results page when JSON is unavailable."""
    hits: List[ReverseHit] = []
    blocks = re.split(r'class="resultadosub"', html, flags=re.I)
    for block in blocks[1:6]:
        sim_m = re.search(r"(\d+(?:\.\d+)?)\s*%", block)
        sim = float(sim_m.group(1)) if sim_m else 0.0
        url_m = re.search(r'href="(https?://[^"]+)"', block)
        ext_url = url_m.group(1) if url_m else ""
        site, site_id = parse_source_url(ext_url)
        if not site:
            if "danbooru" in block.lower():
                dm = re.search(r"danbooru\.donmai\.us/posts/(\d+)", block, re.I)
                if dm:
                    site, site_id = "danbooru", dm.group(1)
            elif "e621" in block.lower():
                em = re.search(r"e621\.net/posts/(\d+)", block, re.I)
                if em:
                    site, site_id = "e621", em.group(1)
        material = ""
        mat_m = re.search(r"Material:\s*([^<\n]+)", block, re.I)
        if mat_m:
            material = mat_m.group(1).strip()
        chars: List[str] = []
        char_m = re.search(r"Characters?:\s*([^<\n]+)", block, re.I)
        if char_m:
            chars = [c.strip() for c in char_m.group(1).split(",") if c.strip()]
        if sim > 0 or site:
            hits.append(
                ReverseHit(
                    similarity=sim,
                    site=site,
                    site_id=site_id,
                    engine="saucenao",
                    material=material,
                    characters=chars,
                    external_url=ext_url,
                    raw={"html_block": block[:500]},
                )
            )
    if not hits:
        for sim_s, url in re.findall(
            r"(\d+(?:\.\d+)?)\s*%[\s\S]{0,400}?href=\"(https?://[^\"]+)\"",
            html,
            flags=re.I,
        )[:5]:
            site, site_id = parse_source_url(url)
            hits.append(
                ReverseHit(
                    similarity=float(sim_s),
                    site=site,
                    site_id=site_id,
                    engine="saucenao",
                    external_url=url,
                    raw={},
                )
            )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def _multipart_post(url: str, field_name: str, filename: str, data: bytes, *, timeout_s: float) -> bytes:
    boundary = "----sdxboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0 (compatible; SDX-ImageProfiler/1.0)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def saucenao_search_web(image_path: str | Path, *, timeout_s: float = 45.0) -> List[ReverseHit]:
    """Upload image to SauceNAO web form — same as the browser, no API key."""
    p = Path(image_path)
    if not p.is_file():
        return []
    _rate_limit("saucenao")
    raw = p.read_bytes()
    # JSON when possible (browser POST + output_type=2).
    for url in (
        "https://saucenao.com/search.php?output_type=2&db=999",
        "https://saucenao.com/search.php?output_type=2&dbmask=1023",
        "https://saucenao.com/search.php",
    ):
        try:
            payload_bytes = _multipart_post(url, "file", p.name, raw, timeout_s=timeout_s)
            text = payload_bytes.decode("utf-8", errors="ignore").strip()
            if text.startswith("{"):
                hits = _hits_from_saucenao_json(json.loads(text))
                if hits:
                    return hits
            if "<html" in text.lower():
                hits = _hits_from_saucenao_html(text)
                if hits:
                    return hits
        except Exception:
            continue
    return []


def saucenao_search_file(
    image_path: str | Path,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
    prefer_web: bool = True,
) -> List[ReverseHit]:
    """Query SauceNAO — web upload by default; API key optional for higher limits."""
    key = (api_key or get_api_key("saucenao")).strip()
    if prefer_web or not key:
        hits = saucenao_search_web(image_path, timeout_s=timeout_s)
        if hits:
            return hits
    if not key:
        return saucenao_search_web(image_path, timeout_s=timeout_s)

    p = Path(image_path)
    if not p.is_file():
        return []
    _rate_limit("saucenao")
    raw = p.read_bytes()
    url = f"https://saucenao.com/search.php?api_key={urllib.parse.quote(key)}&output_type=2&dbmask=1023"
    try:
        payload_bytes = _multipart_post(url, "file", p.name, raw, timeout_s=timeout_s)
        return _hits_from_saucenao_json(json.loads(payload_bytes.decode("utf-8", errors="ignore")))
    except Exception:
        return saucenao_search_web(image_path, timeout_s=timeout_s)


def _hits_from_tineye_json(payload: dict) -> List[ReverseHit]:
    hits: List[ReverseHit] = []
    matches = (payload.get("results") or {}).get("matches")
    if matches is None and isinstance(payload.get("matches"), list):
        matches = payload["matches"]
    for match in matches or []:
        score = float(match.get("score") or 0.0)
        domain = str(match.get("domain") or "")
        image_url = str(match.get("image_url") or "")
        site = ""
        site_id = ""
        ext_url = image_url
        for candidate in _iter_urls_from_tineye_match(match):
            parsed_site, parsed_id = parse_source_url(candidate)
            if parsed_site:
                site, site_id = parsed_site, parsed_id
                ext_url = candidate
                break
        material = domain
        if not site and domain:
            if "danbooru" in domain:
                site = "danbooru"
            elif "e621" in domain:
                site = "e621"
            elif "pixiv" in domain:
                site = "pixiv"
        hits.append(
            ReverseHit(
                similarity=score,
                site=site,
                site_id=site_id,
                engine="tineye",
                material=material,
                external_url=ext_url,
                raw=match,
            )
        )
    hits.sort(key=lambda h: -h.similarity)
    return hits


def tineye_search_web(image_path: str | Path, *, timeout_s: float = 60.0, limit: int = 10) -> List[ReverseHit]:
    """Upload image to TinEye website (same as browser) — no API key."""
    p = Path(image_path)
    if not p.is_file():
        return []
    try:
        import requests
    except ImportError:
        return []

    _rate_limit("tineye")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SDX-ImageProfiler/1.0)"})
    try:
        session.get("https://tineye.com/", timeout=timeout_s)
        raw = p.read_bytes()
        # TinEye SPA uses this internal JSON endpoint after upload.
        for field in ("image", "image_upload"):
            resp = session.post(
                "https://tineye.com/api/v1/result_json/",
                params={"sort": "score", "order": "desc", "limit": limit},
                files={field: (p.name, raw, "application/octet-stream")},
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    hits = _hits_from_tineye_json(payload)
                    if hits:
                        return hits
                except Exception:
                    pass
        # Legacy upload form returns HTML with links.
        resp = session.post(
            "https://tineye.com/search/upload",
            files={"image": (p.name, raw, "application/octet-stream")},
            timeout=timeout_s,
        )
        if resp.status_code == 200 and resp.text:
            hits: List[ReverseHit] = []
            for url in re.findall(r'href="(https?://[^"]+)"', resp.text)[:20]:
                site, site_id = parse_source_url(url)
                if site:
                    hits.append(
                        ReverseHit(
                            similarity=70.0,
                            site=site,
                            site_id=site_id,
                            engine="tineye",
                            external_url=url,
                            raw={},
                        )
                    )
            if hits:
                return hits
    except Exception:
        pass
    return []


def tineye_search_file(
    image_path: str | Path,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 60.0,
    limit: int = 10,
    prefer_web: bool = True,
) -> List[ReverseHit]:
    """Query TinEye — web upload by default; paid API key optional."""
    key = (api_key or get_api_key("tineye")).strip()
    if prefer_web or not key:
        hits = tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)
        if hits:
            return hits
    if not key:
        return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)

    p = Path(image_path)
    if not p.is_file():
        return []

    try:
        import requests
    except ImportError:
        return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)

    _rate_limit("tineye")
    url = "https://api.tineye.com/rest/search/"
    headers = {"x-api-key": key, "User-Agent": "SDX-ImageProfiler/1.0"}
    files = {"image_upload": (p.name, p.read_bytes())}
    data = {"offset": 0, "limit": str(limit), "sort": "score", "order": "desc"}
    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        if int(payload.get("code") or 0) == 200:
            return _hits_from_tineye_json(payload)
    except Exception:
        pass
    return tineye_search_web(image_path, timeout_s=timeout_s, limit=limit)


def reverse_search_file(
    image_path: str | Path,
    *,
    use_saucenao: bool = True,
    use_tineye: bool = True,
    saucenao_api_key: Optional[str] = None,
    tineye_api_key: Optional[str] = None,
) -> List[ReverseHit]:
    """Run SauceNAO and/or TinEye via web upload; merge hits by similarity."""
    hits: List[ReverseHit] = []
    if use_saucenao:
        hits.extend(saucenao_search_file(image_path, api_key=saucenao_api_key))
    if use_tineye:
        hits.extend(tineye_search_file(image_path, api_key=tineye_api_key))
    hits.sort(key=lambda h: (-h.similarity, h.engine))
    return hits


def hit_meets_threshold(hit: ReverseHit, min_sim: float) -> bool:
    """TinEye booru URL matches can use a lower bar — we verify via booru API."""
    if hit.engine == "tineye" and hit.site in ("danbooru", "e621") and hit.site_id:
        return hit.similarity >= min(min_sim, 50.0)
    return hit.similarity >= min_sim
