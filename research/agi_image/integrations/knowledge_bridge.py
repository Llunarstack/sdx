from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class KnowledgeHint:
    """Retrieved fact / rule to merge into conditioning (RAG-style)."""

    source: str
    text: str
    weight: float = 0.35


def merge_knowledge_hints(base_prompt: str, hints: List[KnowledgeHint], max_chars: int = 1200) -> str:
    """Concatenate hints with the raw prompt under a simple character budget."""
    parts: List[str] = []
    budget = max(0, int(max_chars))
    for h in hints:
        chunk = f"[{h.source}] {h.text}".strip()
        if len(chunk) + 2 > budget:
            break
        parts.append(chunk)
        budget -= len(chunk) + 2
    if not parts:
        return base_prompt
    suffix = "; ".join(parts)
    return f"{base_prompt}\n\nContext: {suffix}".strip() if base_prompt.strip() else suffix


__all__ = ["KnowledgeHint", "merge_knowledge_hints"]
