"""Build byte offsets for JSONL lines (seek + partial reads for huge manifests)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def iter_jsonl_line_offsets(path: Path) -> Iterator[Tuple[int, int]]:
    """
    Yield ``(byte_start, byte_length)`` per line including ``\\n`` (if present).

    Uses ``readline`` + ``tell`` — correct for text-mode-safe binary reads.
    """
    with path.open("rb") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            yield (start, len(line))


def build_line_offset_table(path: Path, *, max_lines: Optional[int] = None) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, t in enumerate(iter_jsonl_line_offsets(path)):
        out.append(t)
        if max_lines is not None and len(out) >= max_lines:
            break
    return out


def read_line_at(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        return f.read(length)
