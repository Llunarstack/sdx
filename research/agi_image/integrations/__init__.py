"""Bridging primitives to operational SDX entry points (hints only, safe imports)."""

from .knowledge_bridge import KnowledgeHint, merge_knowledge_hints
from .sample_hints import goal_spec_to_sample_hints

__all__ = ["KnowledgeHint", "goal_spec_to_sample_hints", "merge_knowledge_hints"]
