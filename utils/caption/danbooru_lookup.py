"""Fetch authoritative tags from Danbooru post IDs (after reverse search match)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests

from scripts.scrape.secrets_config import get_credentials, get_secrets_path


@dataclass
class DanbooruPostTags:
    post_id: str
    character_tags: List[str]
    copyright_tags: List[str]
    artist_tags: List[str]
    general_tags: List[str]
    tag_string: str = ""


def _load_creds() -> tuple[Optional[str], Optional[str]]:
    try:
        c = get_credentials("danbooru", get_secrets_path())
        return c.username, c.api_key
    except Exception:
        return None, None


def fetch_danbooru_post(post_id: str, *, timeout_s: float = 30.0) -> Optional[DanbooruPostTags]:
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

    def _split(field: str) -> List[str]:
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
