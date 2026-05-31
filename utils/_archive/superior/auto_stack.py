"""
**Superior prompt stack** — one call to enrich prompts before T5 encode.

Layers: local TF-IDF RAG → domain tips → optional quality negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from utils.prompt.rag_prompt import merge_facts_into_prompt

from .retrieval import TfidfFactIndex, build_tfidf_index_from_jsonl, retrieve_facts_for_query


@dataclass(slots=True)
class SuperiorPromptStack:
    """Configuration for automatic prompt enrichment."""

    rag_jsonl: Optional[str] = None
    rag_top_k: int = 8
    rag_max_chars: int = 4000
    append_domain_tips: bool = True
    _index: Optional[TfidfFactIndex] = field(default=None, repr=False)

    def _get_index(self) -> Optional[TfidfFactIndex]:
        if not self.rag_jsonl:
            return None
        if self._index is None:
            self._index = build_tfidf_index_from_jsonl(self.rag_jsonl)
        return self._index

    def enrich(self, user_prompt: str) -> str:
        """Return prompt with retrieved facts and optional domain hints."""
        prompt = (user_prompt or "").strip()
        idx = self._get_index()
        if idx is not None and idx.n_docs > 0:
            facts = retrieve_facts_for_query(prompt, idx, top_k=self.rag_top_k)
            if facts:
                prompt = merge_facts_into_prompt(prompt, facts, max_chars=self.rag_max_chars)
        if self.append_domain_tips and "plastic skin" not in prompt.lower():
            prompt = prompt + ", natural skin texture, coherent anatomy, sharp focus"
        return prompt.strip()


def apply_superior_prompt_stack(
    user_prompt: str,
    *,
    rag_jsonl: Union[str, Path, None] = None,
    rag_top_k: int = 8,
    append_domain_tips: bool = True,
) -> str:
    """Functional API for scripts and tests."""
    stack = SuperiorPromptStack(
        rag_jsonl=str(rag_jsonl) if rag_jsonl else None,
        rag_top_k=rag_top_k,
        append_domain_tips=append_domain_tips,
    )
    return stack.enrich(user_prompt)


__all__ = ["SuperiorPromptStack", "apply_superior_prompt_stack"]
