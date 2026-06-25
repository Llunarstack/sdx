"""
Lightweight fact/style RAG for prompts — no vector DB required.

Pairs with ``--local-rag-jsonl`` in sample.py; frontier layer adds ranking + injection policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class FactMatch:
    text: str
    score: float
    tags: tuple[str, ...]
    source_line: int


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-z0-9#]+", text.lower()) if len(t) > 2}


class StyleFactRetriever:
    """Overlap-ranked retrieval from JSONL {text, tags?} records."""

    def __init__(self, records: Sequence[dict] | None = None) -> None:
        self.records = list(records or [])

    @classmethod
    def from_jsonl(cls, path: str | Path, *, max_lines: int = 5000) -> "StyleFactRetriever":
        p = Path(path)
        rows: List[dict] = []
        if not p.is_file():
            return cls([])
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"text": line})
        return cls(rows)

    def query(self, prompt: str, *, top_k: int = 3) -> List[FactMatch]:
        q = _tokenize(prompt)
        if not q or not self.records:
            return []
        scored: List[FactMatch] = []
        for i, rec in enumerate(self.records):
            text = str(rec.get("text") or rec.get("fact") or "")
            tags = tuple(str(t) for t in (rec.get("tags") or []))
            cand = _tokenize(text) | _tokenize(" ".join(tags))
            if not cand:
                continue
            overlap = len(q & cand) / max(1, len(q | cand))
            if overlap <= 0:
                continue
            scored.append(FactMatch(text=text, score=overlap, tags=tags, source_line=i))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def inject_prompt(self, prompt: str, matches: Sequence[FactMatch], max_chars: int = 200) -> str:
        if not matches:
            return prompt
        extra: List[str] = []
        n = 0
        for m in matches:
            frag = m.text.strip()
            if not frag:
                continue
            if n + len(frag) > max_chars:
                break
            extra.append(frag)
            n += len(frag)
        if not extra:
            return prompt
        return f"{prompt}, {', '.join(extra)}" if prompt else ", ".join(extra)


__all__ = ["FactMatch", "StyleFactRetriever"]
