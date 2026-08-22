"""Resolve HuggingFace token from env or SDX secrets file."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
_BARE_HF_TOKEN = re.compile(r"^hf_[A-Za-z0-9]{20,}$")
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
        if _BARE_HF_TOKEN.match(line):
            return line
        if in_hf_section and ":" in line:
            k, _, v = line.partition(":")
            if k.strip().lower() in ("token", "api", "key") and v.strip():
                return v.strip()
    return None


def _reject_placeholder(tok: str | None) -> str | None:
    if not tok:
        return None
    if "YOUR_TOKEN" in tok.upper() or tok.endswith("_HERE"):
        return None
    return tok


def hf_token_from_hub() -> str | None:
    """Token from ``huggingface-cli login`` (~/.cache/huggingface/token)."""
    try:
        from huggingface_hub import get_token

        return _reject_placeholder(get_token())
    except Exception:
        return None


def get_hf_token() -> str | None:
    return (
        _reject_placeholder(hf_token_from_env()) or _reject_placeholder(hf_token_from_secrets()) or hf_token_from_hub()
    )


def hf_auth_source() -> str:
    """Where auth came from: env | secret | cli | none."""
    if _reject_placeholder(hf_token_from_env()):
        return "env"
    if _reject_placeholder(hf_token_from_secrets()):
        return "secret"
    if hf_token_from_hub():
        return "cli"
    return "none"


def apply_hf_token_to_env() -> bool:
    tok = get_hf_token()
    if tok:
        os.environ.setdefault("HF_TOKEN", tok)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
        return True
    return False
