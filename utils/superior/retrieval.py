"""
Local **RAG** without external APIs: TF-IDF retrieval over JSONL fact corpora.

Pair with ``utils.prompt.rag_prompt.merge_facts_into_prompt`` to ground generation on your
dataset captions, style notes, or agentic-search exports.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Union

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]{1,}", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass(slots=True)
class TfidfFactIndex:
    """In-memory TF-IDF index over fact strings."""

    facts: List[str]
    doc_freq: Dict[str, int] = field(default_factory=dict)
    doc_tfidf: List[Dict[str, float]] = field(default_factory=list)
    n_docs: int = 0

    def __post_init__(self) -> None:
        self.n_docs = len(self.facts)
        if self.n_docs == 0:
            return
        if not self.doc_tfidf:
            self._build()

    def _build(self) -> None:
        self.doc_freq = {}
        tf_rows: List[Counter[str]] = []
        for fact in self.facts:
            toks = _tokenize(fact)
            c = Counter(toks)
            tf_rows.append(c)
            for tok in c:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
        self.doc_tfidf = []
        for c in tf_rows:
            row: Dict[str, float] = {}
            denom = float(sum(c.values()) or 1)
            for tok, cnt in c.items():
                idf = math.log((1.0 + self.n_docs) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
                row[tok] = (cnt / denom) * idf
            self.doc_tfidf.append(row)

    def query(self, text: str, *, top_k: int = 8) -> List[str]:
        """Return up to ``top_k`` facts ranked by cosine similarity to ``text``."""
        if not self.facts or top_k <= 0:
            return []
        q_toks = _tokenize(text)
        if not q_toks:
            return self.facts[:top_k]
        q_cnt = Counter(q_toks)
        q_denom = float(sum(q_cnt.values()) or 1)
        q_vec: Dict[str, float] = {}
        for tok, cnt in q_cnt.items():
            idf = math.log((1.0 + self.n_docs) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
            q_vec[tok] = (cnt / q_denom) * idf
        scores: List[tuple[float, int]] = []
        for i, dvec in enumerate(self.doc_tfidf):
            dot = sum(q_vec.get(k, 0.0) * v for k, v in dvec.items())
            qn = math.sqrt(sum(v * v for v in q_vec.values())) + 1e-8
            dn = math.sqrt(sum(v * v for v in dvec.values())) + 1e-8
            scores.append((dot / (qn * dn), i))
        scores.sort(key=lambda x: -x[0])
        out: List[str] = []
        seen: set[str] = set()
        for _s, idx in scores[: max(top_k * 2, top_k)]:
            f = self.facts[idx]
            key = f.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= top_k:
                break
        return out


def build_tfidf_index_from_jsonl(
    path: Union[str, Path],
    *,
    text_key: str = "text",
    max_entries: int = 50_000,
    extra_keys: Sequence[str] = ("caption", "summary", "description"),
) -> TfidfFactIndex:
    """Load facts from JSONL (and optional extra string fields per row)."""
    p = Path(path)
    facts: List[str] = []
    if not p.is_file():
        return TfidfFactIndex(facts=[])
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            parts: List[str] = []
            for k in (text_key, *extra_keys):
                v = obj.get(k)
                if v is not None and str(v).strip():
                    parts.append(str(v).strip())
            if parts:
                facts.append(" — ".join(parts))
            if len(facts) >= max_entries:
                break
    return TfidfFactIndex(facts=facts)


def retrieve_facts_for_query(
    query: str,
    index: TfidfFactIndex,
    *,
    top_k: int = 8,
) -> List[str]:
    """Convenience wrapper used by prompt stack and ``sample.py``."""
    return index.query(query, top_k=top_k)


__all__ = [
    "TfidfFactIndex",
    "build_tfidf_index_from_jsonl",
    "retrieve_facts_for_query",
]
