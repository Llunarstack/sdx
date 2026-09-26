"""Fetch authoritative tags from Danbooru post IDs (after reverse search match)."""

from __future__ import annotations

from dataclasses import dataclass

import requests
from scripts.scrape.secrets_config import get_credentials, get_secrets_path


@dataclass
class DanbooruPostTags:
    post_id: str
    character_tags: list[str]
    copyright_tags: list[str]
    artist_tags: list[str]
    general_tags: list[str]
    tag_string: str = ""


def _load_creds() -> tuple[str | None, str | None]:
    try:
        c = get_credentials("danbooru", get_secrets_path())
        return c.username, c.api_key
    except Exception:
        return None, None


def fetch_danbooru_post(post_id: str, *, timeout_s: float = 30.0) -> DanbooruPostTags | None:
    """Return structured tags for a Danbooru post ID (authenticated via secret.txt)."""
    pid = str(post_id).strip()
    if not pid.isdigit():
        return None
    user, api_key = _load_creds()
    params = {}
    if user and api_key:
        params["login"] = user
        params["api_key"] = api_key
    url = f"https://danbooru.donmai.us/posts/{pid}.json"
    try:
        r = requests.get(url, params=params, timeout=timeout_s, headers={"User-Agent": "sdx-image-profiler/1.0"})
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None

    def _split(field: str) -> list[str]:
        raw = d.get(field) or ""
        return [t for t in str(raw).split() if t]

    return DanbooruPostTags(
        post_id=pid,
        character_tags=_split("tag_string_character"),
        copyright_tags=_split("tag_string_copyright"),
        artist_tags=_split("tag_string_artist"),
        general_tags=_split("tag_string_general"),
        tag_string=str(d.get("tag_string") or ""),
    )
