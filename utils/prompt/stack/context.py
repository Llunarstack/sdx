"""Prompt stack context and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class StackMode(str, Enum):
    """Which stages run (inference full, preview subset, training guidance-only)."""

    INFERENCE = "inference"
    PREVIEW = "preview"
    TRAINING = "training"


@dataclass
class PromptArtifacts:
    """Staged fragments produced before the stack runs (layout, multi-instance, …)."""

    layout_negative: str = ""
    multi_instance_negative: str = ""
    detailed_scene_negative: str = ""
    visual_design_negative: str = ""
    character_negative: str = ""
    scene_negative: str = ""
    photo_negative: str = ""  # filled by guidance stage


@dataclass
class PromptContext:
    """
    Mutable input/output carrier for :func:`run_prompt_stack`.

    *positive* / *negative* are updated in place through stages.
    *args* is the argparse Namespace from ``sample.py`` when present.
    """

    positive: str
    negative: str = ""
    mode: StackMode = StackMode.INFERENCE
    args: Any = None
    artifacts: PromptArtifacts = field(default_factory=PromptArtifacts)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace: List[str] = field(default_factory=list)

    # Post-stack hints (sample.py encoding)
    t5_positive_hint: str = ""
    apply_scale_distortion_negative: bool = False


@dataclass
class PromptResult:
    positive: str
    negative: str
    t5_positive_hint: str = ""
    trace: List[str] = field(default_factory=list)
    analysis: Optional[Mapping[str, Any]] = None
    resolved_controls: Optional[Mapping[str, str]] = None
