"""
Bridge semantic understanding to prompt stack and training captions.

Production paths:
  - Layered prompts: ``utils/prompt/prompt_layout.py``, ``--prompt-layout``
  - Regional layout text: ``data/caption_utils.py`` ``[layout]`` blocks
  - Prompt stack: ``utils/prompt/stack/``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .engine import SemanticUnderstandingEngine


def prompt_to_token_tensor(prompt: str, *, max_tokens: int = 64, device: str = "cpu") -> torch.Tensor:
    """Hash tokens into a fixed vocab index tensor for the research decomposer."""
    words = [w for w in (prompt or "").lower().split() if w][:max_tokens]
    if not words:
        return torch.zeros(1, 1, dtype=torch.long, device=device)
    ids = [abs(hash(w)) % 50000 for w in words]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def analyze_prompt_text(prompt: str, engine: Optional[SemanticUnderstandingEngine] = None) -> Dict[str, Any]:
    """Run semantic decomposer on plain text (research / scoring hook)."""
    eng = engine or SemanticUnderstandingEngine()
    tokens = prompt_to_token_tensor(prompt, device="cpu")
    return eng.understand_prompt(tokens)


def layout_sections_from_prompt(prompt: str) -> List[Dict[str, str]]:
    """
    Suggest ``prompt_layout``-style sections from comma clauses (heuristic).

    Returns list of ``{"section": ..., "text": ...}`` for JSON layout authoring.
    """
    clauses = [c.strip() for c in (prompt or "").replace(";", ",").split(",") if c.strip()]
    if not clauses:
        return []
    if len(clauses) == 1:
        return [{"section": "intent", "text": clauses[0]}]
    return [
        {"section": "subjects", "text": clauses[0]},
        *[{"section": "environment", "text": c} for c in clauses[1:-1]],
        {"section": "style", "text": clauses[-1]},
    ]


__all__ = [
    "analyze_prompt_text",
    "layout_sections_from_prompt",
    "prompt_to_token_tensor",
]
