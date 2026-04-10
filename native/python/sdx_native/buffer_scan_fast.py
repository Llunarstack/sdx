"""Fast binary buffer scans (newline count, chunk FNV) without loading whole files into str."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

Buffer = Union[bytes, memoryview]

# Match native_tools.fnv1a64_bytes stream semantics (FNV-1a 64).
_FNV_OFFSET = 146959810393466560
_FNV_PRIME = 1099511628211


def fnv1a64_update(h: int, buf: Buffer) -> int:
    if isinstance(buf, memoryview):
        buf = buf.tobytes()
    x = h
    for b in buf:
        x ^= b
        x = (x * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return x


def count_newlines_buffer(buf: Buffer) -> int:
    if isinstance(buf, memoryview):
        buf = buf.tobytes()
    return buf.count(b"\n")


def scan_file_chunks(
    path: Path,
    *,
    chunk_size: int = 1 << 20,
    hash_seed: int = _FNV_OFFSET,
) -> Tuple[int, int, int]:
    """
    Stream ``path`` in binary: return ``(fnv1a64_hash, newline_count, byte_count)``.

    Semantics align with :func:`sdx_native.native_tools.fnv1a64_file` for hashing and counts.
    """
    h = hash_seed & 0xFFFFFFFFFFFFFFFF
    nlines = 0
    nbytes = 0
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            nbytes += len(block)
            nlines += block.count(b"\n")
            h = fnv1a64_update(h, block)
    return h, nlines, nbytes
