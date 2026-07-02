"""Parse the free-form credentials file into per-site login/api-key config.

The secrets file is informal (``key: value`` lines under a site header). This
loader is tolerant of the exact spellings used in ``D:\\Development\\secret.txt``.
Credentials are only ever read into memory here; nothing is written back or
logged. Keep the secrets file OUTSIDE the repo so it is never committed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_SECRETS_PATH = r"D:\Development\secret.txt"

# Map the various header spellings in the file to canonical site keys.
_SITE_ALIASES = {
    "danbooru": "danbooru",
    "e621": "e621",
    "rule34xxx": "rule34xxx",
    "rule34.xxx": "rule34xxx",
    "rule34.xyz": "rule34xyz",
}

_USER_KEYS = ("user", "username", "login")
_PASS_KEYS = ("pas", "pass", "password", "p")
_API_KEYS = ("api", "api_key", "apikey", "api?", "key")
_EMAIL_KEYS = ("e", "em", "email")


@dataclass
class SiteCredentials:
    site: str
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None


def _extract_rule34_apikey_userid(value: str) -> tuple[Optional[str], Optional[str]]:
    """Pull api_key/user_id out of a raw ``&api_key=...&user_id=...`` blob."""
    api = re.search(r"api_key=([0-9a-fA-F]+)", value)
    uid = re.search(r"user_id=(\d+)", value)
    return (api.group(1) if api else None, uid.group(1) if uid else None)


def parse_secrets_file(path: str | os.PathLike[str] | None = None) -> dict[str, SiteCredentials]:
    """Return ``{canonical_site: SiteCredentials}`` parsed from the secrets file."""
    p = Path(path or os.environ.get("SDX_SECRETS_FILE") or DEFAULT_SECRETS_PATH)
    if not p.is_file():
        raise FileNotFoundError(
            f"Secrets file not found: {p}. Pass --secrets PATH or set SDX_SECRETS_FILE."
        )

    out: dict[str, SiteCredentials] = {}
    current: Optional[SiteCredentials] = None

    for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header = line.rstrip(":").strip().lower()
        if header in _SITE_ALIASES and (":" not in line or line.endswith(":") or " " not in line.rstrip(":")):
            site = _SITE_ALIASES[header]
            current = out.setdefault(site, SiteCredentials(site=site))
            continue

        if ":" not in line or current is None:
            continue

        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue

        if current.site == "rule34xxx" and "api_key=" in value:
            api, uid = _extract_rule34_apikey_userid(value)
            current.api_key = current.api_key or api
            current.user_id = current.user_id or uid
            continue

        if key in _USER_KEYS:
            current.username = current.username or value
        elif key in _PASS_KEYS:
            current.password = current.password or value
        elif key in _API_KEYS:
            current.api_key = current.api_key or value
        elif key in _EMAIL_KEYS:
            current.email = current.email or value
        elif key in ("ip",) and current.site == "danbooru" and current.api_key is None:
            # danbooru api keys are sometimes mislabeled in the file
            current.api_key = value

    return out


def get_credentials(site: str, path: str | os.PathLike[str] | None = None) -> SiteCredentials:
    creds = parse_secrets_file(path)
    canonical = _SITE_ALIASES.get(site.lower(), site.lower())
    if canonical not in creds:
        raise KeyError(f"No credentials for site {site!r} in secrets file. Found: {sorted(creds)}")
    return creds[canonical]
