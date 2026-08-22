"""rule34.xyz v2 JSON API (replaced legacy Gelbooru ``page=dapi`` in 2025)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import requests

from .booru_client import Post, RateLimiter
from .post_cap import post_cap_reached, posts_remaining
from .secrets_config import SiteCredentials

ROOT = "https://rule34.xyz"
ROOT_CDN = "https://rule34xyz.b-cdn.net"
SEARCH_URL = f"{ROOT}/api/v2/post/search/root"
SIGNIN_URL = f"{ROOT}/api/v2/auth/signin"

# Prefer video for animated posts, then still image preview.
_FORMAT_PRIORITY = ("100", "101", "102", "10", "11", "12", "13", "14", "31", "32", "112")
_FORMAT_EXT = {
    "10": ("pic.jpg", "jpg"),
    "11": ("pic.jpg", "jpg"),
    "12": ("pic.jpg", "jpg"),
    "13": ("pic.jpg", "jpg"),
    "14": ("pic.jpg", "jpg"),
    "31": ("pic.jpg", "jpg"),
    "32": ("pic.jpg", "jpg"),
    "100": ("mov.mp4", "mp4"),
    "101": ("mov720.mp4", "mp4"),
    "102": ("mov480.mp4", "mp4"),
    "112": ("mov.mp4", "mp4"),
}
_TAG_TYPES = {
    2: "copyright",
    4: "character",
    8: "artist",
}


def _md5_for_post(post_id: int | str) -> str:
    return hashlib.md5(f"rule34xyz:{post_id}".encode()).hexdigest()


def login(session: requests.Session, creds: SiteCredentials, *, timeout_s: float = 45.0) -> None:
    """Attach ``Authorization: Bearer …`` when email/password or token is available."""
    if creds.api_key and str(creds.api_key).lower().startswith("bearer "):
        session.headers["Authorization"] = creds.api_key
        return
    if creds.api_key and "." in creds.api_key:
        session.headers["Authorization"] = f"Bearer {creds.api_key}"
        return

    email = (creds.email or creds.username or "").strip()
    password = (creds.password or "").strip()
    if not email or not password:
        return  # anonymous search still works for many tags

    r = session.post(SIGNIN_URL, json={"email": email, "password": password}, timeout=timeout_s)
    r.raise_for_status()
    jwt = (r.json() or {}).get("jwt")
    if not jwt:
        raise RuntimeError(f"{creds.site}: sign-in succeeded but no jwt returned")
    session.headers["Authorization"] = f"Bearer {jwt}"


def file_url_from_post(post: dict) -> tuple[str, str]:
    files = post.get("files") or {}
    fmt = None
    for key in _FORMAT_PRIORITY:
        if key in files:
            fmt = key
            break
    if fmt is None and files:
        fmt = next(iter(files))
    if fmt is None:
        return "", ""

    filename, ext = _FORMAT_EXT.get(fmt, (f"file.{fmt}", "bin"))
    post_id = int(post["id"])
    use_cdn = bool((files.get(fmt) or [0])[0])
    root = ROOT_CDN if use_cdn else ROOT
    url = f"{root}/posts/{post_id // 1000}/{post_id}/{post_id}.{filename}"
    return url, ext


def tags_from_post(post: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    all_tags: list[str] = []
    artists: list[str] = []
    characters: list[str] = []
    copyrights: list[str] = []
    for tag in post.get("tags") or []:
        if not isinstance(tag, dict):
            continue
        value = str(tag.get("value", "")).strip()
        if not value:
            continue
        all_tags.append(value)
        bucket = _TAG_TYPES.get(int(tag.get("type", 0) or 0))
        if bucket == "artist":
            artists.append(value)
        elif bucket == "character":
            characters.append(value)
        elif bucket == "copyright":
            copyrights.append(value)
    return all_tags, artists, characters, copyrights


def post_from_api(post: dict, *, site: str = "rule34xyz") -> Post | None:
    file_url, ext = file_url_from_post(post)
    if not file_url:
        return None
    post_id = str(post.get("id", ""))
    tags, artists, characters, copyrights = tags_from_post(post)
    return Post(
        site=site,
        id=post_id,
        md5=_md5_for_post(post_id),
        file_url=file_url,
        ext=ext,
        tags=tags,
        rating="",
        width=int(post.get("width", 0) or 0),
        height=int(post.get("height", 0) or 0),
        artist_tags=artists,
        character_tags=characters,
        copyright_tags=copyrights,
    )


class Rule34xyzV2Adapter:
    """rule34.xyz adapter using ``/api/v2/post/search/root`` (POST JSON)."""

    site = "rule34xyz"
    posts_url = SEARCH_URL
    first_page = 0
    page_limit = 60
    cursor_supported = False
    auth = None

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds

    def build_params(self, tags: str, page: int, before_id: int | None = None) -> dict:
        raise NotImplementedError("rule34xyz uses iter_posts()")

    def parse_posts(self, data) -> Iterator[Post]:
        return iter(())

    def iter_posts(
        self,
        session: requests.Session,
        tags: str,
        *,
        max_posts: int,
        limiter: RateLimiter,
        timeout_s: float,
    ) -> Iterator[Post]:
        login(session, self.creds, timeout_s=timeout_s)
        include_tags = [t for t in tags.split() if t.strip()] if tags else []
        skip = 0
        cursor: str | None = None
        yielded = 0
        per_page = self.page_limit

        while not post_cap_reached(yielded, max_posts):
            limiter.wait()
            take = per_page if max_posts <= 0 else min(per_page, posts_remaining(max_posts, yielded) or per_page)
            body: dict = {
                "Skip": skip,
                "take": take,
                "CountTotal": False,
                "IncludeLinks": True,
                "OrderBy": 0,
            }
            if include_tags:
                body["includeTags"] = include_tags
            if cursor:
                body["cursor"] = cursor

            r = session.post(SEARCH_URL, json=body, timeout=timeout_s)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} from rule34.xyz search", response=r)
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("items") or []
            if not items:
                break

            for item in items:
                if post_cap_reached(yielded, max_posts):
                    return
                post_data = item
                if not post_data.get("tags"):
                    limiter.wait()
                    pr = session.get(f"{ROOT}/api/v2/post/{item['id']}", timeout=timeout_s)
                    pr.raise_for_status()
                    post_data = pr.json()
                post = post_from_api(post_data)
                if post is None:
                    continue
                yield post
                yielded += 1

            if len(items) < per_page:
                break
            skip += len(items)
            cursor = data.get("cursor")
