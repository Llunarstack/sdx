"""Resolve HuggingFace token from env or SDX secrets file."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_TOKEN_LINE = re.compile(
    r"^(?:hf[_-]?token|huggingface[_-]?token|token)\s*[:=]\s*(\S+)",
    re.IGNORECASE,
)


def hf_token_from_env() -> str | None:
    for key in _ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def hf_token_from_secrets(path: str | os.PathLike[str] | None = None) -> str | None:
    p = Path(path or os.environ.get("SDX_SECRETS_FILE", "/workspace/secret.txt"))
    if not p.is_file():
        return None
    in_hf_section = False
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head = line.rstrip(":").strip().lower()
        if head in ("huggingface", "hf", "hugging face"):
            in_hf_section = True
            continue
        if in_hf_section and ":" not in line and not _TOKEN_LINE.match(line):
            in_hf_section = False
        m = _TOKEN_LINE.match(line)
        if m:
            return m.group(1).strip()
        if in_hf_section and ":" in line:
            k, _, v = line.partition(":")
            if k.strip().lower() in ("token", "api", "key") and v.strip():
                return v.strip()
    return None


def get_hf_token() -> str | None:
    return hf_token_from_env() or hf_token_from_secrets()


def apply_hf_token_to_env() -> bool:
    tok = get_hf_token()
    if tok:
        os.environ.setdefault("HF_TOKEN", tok)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
        return True
    return False
