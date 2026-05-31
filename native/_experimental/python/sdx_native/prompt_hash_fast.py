"""Fast stable hashes for prompt / caption cache keys (avoid heavy str work)."""

from __future__ import annotations

import hashlib

from sdx_native.caption_csv_fast import normalize_caption_csv


def blake2b_key(text: str, *, digest_size: int = 16) -> bytes:
    """Short Blake2b digest of UTF-8 bytes (good for cache dict keys)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).digest()


def blake2b_hex(text: str, *, digest_size: int = 16) -> str:
    return blake2b_key(text, digest_size=digest_size).hex()


def normalized_caption_key(text: str, *, digest_size: int = 16) -> str:
    """Normalize CSV caption then hash (dedupe-friendly)."""
    return blake2b_hex(normalize_caption_csv(text), digest_size=digest_size)


def try_xxhash_hex(text: str) -> str | None:
    """Return xxh64 hex if ``xxhash`` is installed, else ``None``."""
    try:
        import xxhash  # type: ignore

        return xxhash.xxh64(text.encode("utf-8")).hexdigest()
    except Exception:
        return None
