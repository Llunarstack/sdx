"""Per-site API adapters. Each normalizes its API into :class:`Post` objects.

APIs used (all official JSON endpoints):
  danbooru   https://danbooru.donmai.us/posts.json      (login + api_key)
  e621       https://e621.net/posts.json                (HTTP basic: login + api_key)
  rule34xxx  https://api.rule34.xxx/index.php?...&json=1 (api_key + user_id)
  rule34xyz  https://rule34.xyz/api/v2/post/search/root   (JWT from email/password)
"""

from __future__ import annotations

from typing import Iterator, Optional

from .booru_client import Post
from .rule34xyz_v2 import Rule34xyzV2Adapter
from .safety import blocked_query_tags
from .secrets_config import SiteCredentials


class DanbooruAdapter:
    site = "danbooru"
    posts_url = "https://danbooru.donmai.us/posts.json"
    first_page = 1
    # danbooru limits anonymous/basic accounts to 2 search tags, so we cannot
    # inject the blocklist here; the per-post gate in BooruClient is authoritative.
    page_limit = 200
    # Numeric paging caps at ~page 1000; id-cursor (page=b<id>) crawls the whole site.
    cursor_supported = True

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        self.auth: Optional[tuple[str, str]] = None

    def build_params(self, tags: str, page: int, before_id: Optional[int] = None) -> dict:
        params = {"tags": tags, "limit": self.page_limit}
        params["page"] = f"b{before_id}" if before_id else page
        if self.creds.username and self.creds.api_key:
            params["login"] = self.creds.username
            params["api_key"] = self.creds.api_key
        return params

    def parse_posts(self, data) -> Iterator[Post]:
        if not isinstance(data, list):
            return
        for d in data:
            file_url = d.get("file_url") or d.get("large_file_url") or ""
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=d.get("md5", ""),
                file_url=file_url,
                ext=d.get("file_ext", "") or _ext_from_url(file_url),
                tags=(d.get("tag_string", "") or "").split(),
                rating=d.get("rating", ""),
                width=int(d.get("image_width", 0) or 0),
                height=int(d.get("image_height", 0) or 0),
                artist_tags=_split_tag_field(d.get("tag_string_artist")),
                character_tags=_split_tag_field(d.get("tag_string_character")),
                copyright_tags=_split_tag_field(d.get("tag_string_copyright")),
            )


class E621Adapter:
    site = "e621"
    posts_url = "https://e621.net/posts.json"
    first_page = 1
    page_limit = 320  # e621 hard max
    cursor_supported = True  # page=b<id> crawls past the deep-page cap

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        # e621 accepts credentials via HTTP basic auth.
        self.auth: Optional[tuple[str, str]] = (
            (creds.username, creds.api_key) if creds.username and creds.api_key else None
        )

    def build_params(self, tags: str, page: int, before_id: Optional[int] = None) -> dict:
        # e621 allows up to 40 tags, so we can negate the blocklist in-query too.
        full_tags = " ".join([tags, *blocked_query_tags()]).strip()
        params = {"tags": full_tags, "limit": self.page_limit}
        params["page"] = f"b{before_id}" if before_id else page
        return params

    def parse_posts(self, data) -> Iterator[Post]:
        posts = data.get("posts") if isinstance(data, dict) else None
        if not isinstance(posts, list):
            return
        for d in posts:
            f = d.get("file") or {}
            tag_groups = d.get("tags") or {}
            all_tags: list[str] = []
            for group in tag_groups.values():
                if isinstance(group, list):
                    all_tags.extend(group)
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=f.get("md5", ""),
                file_url=f.get("url") or "",
                ext=f.get("ext", ""),
                tags=all_tags,
                rating=d.get("rating", ""),
                width=int(f.get("width", 0) or 0),
                height=int(f.get("height", 0) or 0),
                artist_tags=list(tag_groups.get("artist") or []),
                character_tags=list(tag_groups.get("character") or []),
                copyright_tags=list(tag_groups.get("copyright") or []),
            )


def _iter_gelbooru_posts(data, *, site: str, cdn_base: str) -> Iterator[Post]:
    """Parse Gelbooru / rule34 JSON (array or ``{"post": [...]}`` wrapper)."""
    posts = None
    if isinstance(data, list):
        posts = data
    elif isinstance(data, dict):
        posts = data.get("post")
        if isinstance(posts, dict):
            posts = [posts]
    if not posts:
        return
    for d in posts:
        if not isinstance(d, dict):
            continue
        file_url = d.get("file_url") or ""
        if not file_url and d.get("directory") is not None and d.get("image"):
            file_url = f"{cdn_base.rstrip('/')}/images/{d['directory']}/{d['image']}"
        yield Post(
            site=site,
            id=str(d.get("id", "")),
            md5=d.get("hash", "") or d.get("md5", ""),
            file_url=file_url,
            ext=_ext_from_url(file_url),
            tags=(d.get("tags", "") or "").split(),
            rating=d.get("rating", ""),
            width=int(d.get("width", 0) or 0),
            height=int(d.get("height", 0) or 0),
        )


class _GelbooruDapiAdapter:
    """Shared Gelbooru ``page=dapi&s=post&q=index`` adapter."""

    first_page = 0
    page_limit = 1000
    cursor_supported = False
    cdn_base = ""

    def __init__(self, creds: SiteCredentials, *, posts_url: str, site: str, cdn_base: str) -> None:
        self.creds = creds
        self.posts_url = posts_url
        self.site = site
        self.cdn_base = cdn_base
        self.auth = None

    def build_params(self, tags: str, page: int, before_id: Optional[int] = None) -> dict:
        params = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "tags": tags,
            "pid": page,
            "limit": self.page_limit,
        }
        if self.creds.api_key and self.creds.user_id:
            params["api_key"] = self.creds.api_key
            params["user_id"] = self.creds.user_id
        return params

    def parse_posts(self, data) -> Iterator[Post]:
        return _iter_gelbooru_posts(data, site=self.site, cdn_base=self.cdn_base)


class Rule34xxxAdapter(_GelbooruDapiAdapter):
    site = "rule34xxx"

    def __init__(self, creds: SiteCredentials) -> None:
        super().__init__(
            creds,
            posts_url="https://api.rule34.xxx/index.php",
            site=self.site,
            cdn_base="https://api-cdn.rule34.xxx",
        )


class Rule34xyzAdapter(Rule34xyzV2Adapter):
    site = "rule34xyz"


ADAPTERS = {
    "danbooru": DanbooruAdapter,
    "e621": E621Adapter,
    "rule34xxx": Rule34xxxAdapter,
    "rule34xyz": Rule34xyzAdapter,
}

# Default scrape set for RunPod image training (not e621 / rule34.xyz).
DEFAULT_SCRAPE_SITES = ("danbooru", "rule34xxx")

# API pagination rate limits (req/s). Image CDN downloads use a separate thread
# pool and do NOT consume this budget.
#   danbooru: 10 reads/s global cap (donmai.us wiki) — stay at 8 for headroom
#   rule34xxx: api.rule34.xxx DAPI; Gelbooru-family accounts ~10/s with creds
#   e621: hard 2/s — never exceed 1.5
#   rule34xyz: JWT v2 API — keep conservative
RATE_LIMITS = {
    "danbooru": 9.5,
    "e621": 1.5,
    "rule34xxx": 9.5,
    "rule34xyz": 2.0,
}


def _ext_from_url(url: str) -> str:
    if "." in url.rsplit("/", 1)[-1]:
        return url.rsplit(".", 1)[-1].split("?")[0].lower()
    return ""


def _split_tag_field(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [t for t in str(value).split() if t.strip()]


def build_adapter(site: str, creds: SiteCredentials):
    site = site.lower()
    if site not in ADAPTERS:
        raise ValueError(f"Unsupported site {site!r}. Choices: {sorted(ADAPTERS)}")
    return ADAPTERS[site](creds)
