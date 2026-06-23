"""
Cross-attention layout guidance plan (training-free).

Maps box regions to *when* and *how strongly* to enforce spatial attention —
hook point for future DiT cross-attn processors (Dense Diffusion / BoxDiff lineage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class AttentionLayoutPlan:
    """Per-region attention enforcement schedule."""

    region_names: Tuple[str, ...]
    boxes: Tuple[Tuple[float, float, float, float], ...]
    enforce_steps: Tuple[int, ...]  # step indices to apply hard masking
    strength: float = 0.85
    backward_guidance: bool = True  # prefer backward over forward (arXiv:2304.03373)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_names": list(self.region_names),
            "boxes": [list(b) for b in self.boxes],
            "enforce_steps": list(self.enforce_steps),
            "strength": self.strength,
            "backward_guidance": self.backward_guidance,
        }


def build_attention_layout_plan(
    regions: Sequence[Any],
    *,
    num_steps: int = 28,
    inject_frac: float = 0.4,
    strength: float = 0.85,
) -> AttentionLayoutPlan:
    """
    Build a plan from box-layout regions (objects with ``name``, ``x1``…``y2``).

    Enforces layout primarily in the first ``inject_frac`` of steps (structure phase).
    """
    names: List[str] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for r in regions:
        names.append(str(getattr(r, "name", f"region_{len(names)}")))
        boxes.append(
            (
                float(getattr(r, "x1", 0)),
                float(getattr(r, "y1", 0)),
                float(getattr(r, "x2", 1)),
                float(getattr(r, "y2", 1)),
            )
        )
    n_enforce = max(1, int(round(num_steps * inject_frac)))
    enforce = tuple(range(n_enforce))
    return AttentionLayoutPlan(
        region_names=tuple(names),
        boxes=tuple(boxes),
        enforce_steps=enforce,
        strength=float(strength),
        backward_guidance=True,
    )
