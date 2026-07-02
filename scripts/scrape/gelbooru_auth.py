"""Bootstrap Gelbooru-style API credentials (api_key + user_id) from a login.

rule34.xyz and similar forks often only have email/password in the secrets file.
This logs in once, scrapes the account options page, and returns credentials for
the standard ``page=dapi`` JSON API.
"""

from __future__ import annotations

import re
from typing import Optional

import requests

from .secrets_config import SiteCredentials


def _parse_options_page(html: str) -> tuple[Optional[str], Optional[str]]:
    api_key = None
    user_id = None
    m = re.search(r'name=["\']api_key["\'][^>]*value=["\']([^"\']+)["\']', html, re.I)
    if m:
        api_key = m.group(1).strip()
    m = re.search(r'name=["\']user_id["\'][^>]*value=["\'](\d+)["\']', html, re.I)
    if m:
        user_id = m.group(1).strip()
    if not api_key:
        m = re.search(r"api[_\s-]?key['\"]?\s*[:=]\s*['\"]?([0-9a-f]{16,})", html, re.I)
        if m:
            api_key = m.group(1).strip()
    if not user_id:
        m = re.search(r"user[_\s-]?id['\"]?\s*[:=]\s*['\"]?(\d+)", html, re.I)
        if m:
            user_id = m.group(1).strip()
    return api_key, user_id


def bootstrap_gelbooru_credentials(
    base_url: str,
    creds: SiteCredentials,
    *,
    session: Optional[requests.Session] = None,
    timeout_s: float = 45.0,
) -> SiteCredentials:
    """Return creds with api_key/user_id filled in (login if needed)."""
    if creds.api_key and creds.user_id:
        return creds

    user = (creds.username or creds.email or "").strip()
    password = (creds.password or "").strip()
    if not user or not password:
        raise ValueError(
            f"{creds.site}: Gelbooru API needs api_key+user_id or username/email+password in the secrets file."
        )

    sess = session or requests.Session()
    root = base_url.rstrip("/")
    index = f"{root}/index.php"

    login = sess.post(
        index,
        params={"page": "account", "s": "login", "code": "00"},
        data={"user": user, "pass": password},
        timeout=timeout_s,
        allow_redirects=True,
    )
    login.raise_for_status()

    opts = sess.get(index, params={"page": "account", "s": "options"}, timeout=timeout_s)
    opts.raise_for_status()
    api_key, user_id = _parse_options_page(opts.text)

    if not api_key or not user_id:
        raise RuntimeError(
            f"{creds.site}: logged in but could not parse api_key/user_id from account options. "
            "Add them manually to the secrets file."
        )

    return SiteCredentials(
        site=creds.site,
        username=creds.username or user,
        password=creds.password,
        api_key=api_key,
        user_id=user_id,
        email=creds.email,
    )
