"""Load optional API keys; reverse search defaults to free web upload."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=8)
def get_api_key(service: str, secrets_path: Optional[str] = None) -> str:
    """Optional API key for SauceNAO/TinEye (faster limits). Web upload works without one."""
    svc = service.strip().lower()
    env_name = f"{svc.upper()}_API_KEY"
    from_env = (os.environ.get(env_name) or "").strip()
    if from_env:
        return from_env
    try:
        from scripts.scrape.secrets_config import get_credentials

        creds = get_credentials(svc, secrets_path)
        return (creds.api_key or "").strip()
    except Exception:
        return ""


def reverse_search_enabled(*, use_saucenao: bool = True, use_tineye: bool = True) -> bool:
    """Web image upload works without API keys — enabled whenever caller wants reverse search."""
    return bool(use_saucenao or use_tineye)
