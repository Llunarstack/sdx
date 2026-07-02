"""Per-site API adapters. Each normalizes its API into :class:`Post` objects.

APIs used (all official JSON endpoints):
  danbooru   https://danbooru.donmai.us/posts.json      (login + api_key)
  e621       https://e621.net/posts.json                (HTTP basic: login + api_key)
  rule34xxx  https://api.rule34.xxx/index.php?...&json=1 (api_key + user_id)
"""

from __future__ import annotations

from typing import Iterator, Optional

from .booru_client import Post
from .safety import blocked_query_tags
from .secrets_config import SiteCredentials


class DanbooruAdapter:
    site = "danbooru"
    posts_url = "https://danbooru.donmai.us/posts.json"
    first_page = 1
    # danbooru limits anonymous/basic accounts to 2 search tags, so we cannot
    # inject the blocklist here; the per-post gate in BooruClient is authoritative.
    page_limit = 200

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        self.auth: Optional[tuple[str, str]] = None

    def build_params(self, tags: str, page: int) -> dict:
        params = {"tags": tags, "page": page, "limit": self.page_limit}
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
            )


class E621Adapter:
    site = "e621"
    posts_url = "https://e621.net/posts.json"
    first_page = 1
    page_limit = 320  # e621 hard max

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        # e621 accepts credentials via HTTP basic auth.
        self.auth: Optional[tuple[str, str]] = (
            (creds.username, creds.api_key) if creds.username and creds.api_key else None
        )

    def build_params(self, tags: str, page: int) -> dict:
        # e621 allows up to 40 tags, so we can negate the blocklist in-query too.
        full_tags = " ".join([tags, *blocked_query_tags()]).strip()
        return {"tags": full_tags, "page": page, "limit": self.page_limit}

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
            )


class Rule34xxxAdapter:
    site = "rule34xxx"
    posts_url = "https://api.rule34.xxx/index.php"
    first_page = 0  # rule34 pages are 0-indexed (pid)
    page_limit = 1000

    def __init__(self, creds: SiteCredentials) -> None:
        self.creds = creds
        self.auth = None

    def build_params(self, tags: str, page: int) -> dict:
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
        # rule34 returns a JSON array, or an empty string / None past the end.
        if not isinstance(data, list):
            return
        for d in data:
            file_url = d.get("file_url") or ""
            if not file_url and d.get("directory") is not None and d.get("image"):
                file_url = f"https://api-cdn.rule34.xxx/images/{d['directory']}/{d['image']}"
            yield Post(
                site=self.site,
                id=str(d.get("id", "")),
                md5=d.get("hash", ""),
                file_url=file_url,
                ext=_ext_from_url(file_url),
                tags=(d.get("tags", "") or "").split(),
                rating=d.get("rating", ""),
                width=int(d.get("width", 0) or 0),
                height=int(d.get("height", 0) or 0),
            )


ADAPTERS = {
    "danbooru": DanbooruAdapter,
    "e621": E621Adapter,
    "rule34xxx": Rule34xxxAdapter,
}

# Polite per-site request ceilings (requests/sec). e621 enforces <= 2/s hard.
RATE_LIMITS = {
    "danbooru": 4.0,
    "e621": 1.5,
    "rule34xxx": 2.0,
}


def _ext_from_url(url: str) -> str:
    if "." in url.rsplit("/", 1)[-1]:
        return url.rsplit(".", 1)[-1].split("?")[0].lower()
    return ""


def build_adapter(site: str, creds: SiteCredentials):
    site = site.lower()
    if site not in ADAPTERS:
        raise ValueError(f"Unsupported site {site!r}. Choices: {sorted(ADAPTERS)}")
    return ADAPTERS[site](creds)
