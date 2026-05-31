"""
**Visual Brain** — understand references, search, dissect, plan, generate, edit, verify.

Orchestrates OCR/VLM understanding, web image search, dissection, ControlNet maps,
and iterative generation/editing while keeping the user prompt as the source of truth.
"""

from __future__ import annotations

from .scene_brief import SceneBrief, SceneElement
from .visual_brain import VisualBrain, VisualBrainConfig, VisualBrainResult

__all__ = [
    "SceneBrief",
    "SceneElement",
    "VisualBrain",
    "VisualBrainConfig",
    "VisualBrainResult",
]
