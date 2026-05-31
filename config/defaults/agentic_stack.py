"""
Defaults for the **Agentic Stack** (``utils/agentic/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class AgenticStackDefaults:
    max_reflect_loops: int = 3
    num_candidates: int = 4
    pick_metric: str = "superior_composite"
    expand_prompt: bool = True
    self_correct: bool = True
    rag_top_k: int = 8
    min_composite_accept: float = 0.62
    min_clip_accept: float = 0.22
    qwen_path: str = ""
    trajectory_variants: int = 3
    evolve_iterations: int = 2
    superior_preset: str = "superior"
    extra_sample_args: List[str] = field(
        default_factory=lambda: [
            "--zeresfdg-strength",
            "1",
            "--cfg-zero-star",
            "--qsilk-micrograin",
            "0.1",
            "--cfg-rescale",
            "0.7",
            "--human-made",
            "standard",
        ]
    )


DEFAULTS = AgenticStackDefaults()

__all__ = ["AgenticStackDefaults", "DEFAULTS"]
