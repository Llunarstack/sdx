"""Fetch authoritative tags from e621 post IDs (after reverse search match)."""

from __future__ import annotations

from dataclasses import dataclass

import requests
from scripts.scrape.secrets_config import get_credentials, get_secrets_path


@dataclass
class E621PostTags:
    post_id: str
    character_tags: list[str]
    copyright_tags: list[str]
    artist_tags: list[str]
    general_tags: list[str]
    tag_string: str = ""


def _auth() -> tuple[str, str] | None:
    try:
        c = get_credentials("e621", get_secrets_path())
        if c.username and c.api_key:
            return c.username, c.api_key
    except Exception:
        pass
    return None


def fetch_e621_post(post_id: str, *, timeout_s: float = 30.0) -> E621PostTags | None:
    pid = str(post_id).strip()
    if not pid.isdigit():
        return None
    url = f"https://e621.net/posts/{pid}.json"
    auth = _auth()
    headers = {"User-Agent": "sdx-image-profiler/1.0"}
    try:
        r = requests.get(url, auth=auth, timeout=timeout_s, headers=headers)
        r.raise_for_status()
        d = r.json().get("post") or {}
    except Exception:
        return None

    tags = d.get("tags") or {}

    def _lst(key: str) -> list[str]:
        val = tags.get(key) or []
        return [str(x) for x in val if str(x).strip()]

    all_tags = []
    for group in tags.values():
        if isinstance(group, list):
            all_tags.extend(str(x) for x in group if str(x).strip())

    return E621PostTags(
        post_id=pid,
        character_tags=_lst("character"),
        copyright_tags=_lst("copyright"),
        artist_tags=_lst("artist"),
        general_tags=_lst("general"),
        tag_string=" ".join(all_tags),
    )
