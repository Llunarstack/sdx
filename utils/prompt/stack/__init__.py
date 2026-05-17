"""
Unified prompt stack for SDX.

Use :func:`run_prompt_stack` with a :class:`PromptContext`, or
:func:`apply_sample_prompt_stack` from ``sample.py``.

Training captions can reuse :func:`merge_guidance_for_training_caption`.
"""

from __future__ import annotations

from .clauses import CLAUSE_REGISTRY, IntentClause, apply_clauses, list_clauses
from .context import PromptArtifacts, PromptContext, PromptResult, StackMode
from .controls import ContentControlState, merge_content_control_overrides, resolve_content_controls
from .intelligence import PromptAnalysis, analyze_prompt
from .runner import run_prompt_stack
from .sample_bridge import apply_sample_prompt_stack
from .stages.guidance import apply_training_guidance_pair, merge_guidance_for_training_caption
from .tokens import append_unique, join_tags, merge_fragments, split_tags

__all__ = [
    "CLAUSE_REGISTRY",
    "ContentControlState",
    "IntentClause",
    "PromptAnalysis",
    "PromptArtifacts",
    "PromptContext",
    "PromptResult",
    "StackMode",
    "analyze_prompt",
    "apply_clauses",
    "apply_sample_prompt_stack",
    "apply_training_guidance_pair",
    "merge_content_control_overrides",
    "append_unique",
    "join_tags",
    "list_clauses",
    "merge_fragments",
    "merge_guidance_for_training_caption",
    "resolve_content_controls",
    "run_prompt_stack",
    "split_tags",
]
