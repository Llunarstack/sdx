"""HYDRA-SLOTS — Lightweight multi-entity slot prior (prompt + box only).

Novel stills idea: treat each counted entity as a soft 'slot' with its own
prompt shard and non-overlapping bbox, without requiring trained slot attention.
Feeds regional CFG / layout attn already in SDX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from utils.generation.inventions.countgate import count_grid_boxes, plan_countgate

__all__ = ["HydraSlot", "HydraSlotPlan", "plan_hydra_slots"]


@dataclass
class HydraSlot:
    slot_id: str
    prompt: str
    box: list[float]


@dataclass
class HydraSlotPlan:
    slots: list[HydraSlot] = field(default_factory=list)
    global_prompt: str = ""
    box_layout: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [asdict(s) for s in self.slots],
            "global_prompt": self.global_prompt,
            "box_layout": self.box_layout,
            "notes": list(self.notes),
        }


def plan_hydra_slots(prompt: str, *, max_slots: int = 6) -> HydraSlotPlan:
    cg = plan_countgate(prompt)
    n = 0
    noun = "subject"
    if cg.specs:
        primary = max(cg.specs, key=lambda s: s.n)
        n = min(int(primary.n), int(max_slots))
        noun = primary.noun
    if n < 2:
        return HydraSlotPlan(global_prompt=prompt, notes=["hydra inactive (<2 count)"])

    boxes = cg.boxes or count_grid_boxes(n)
    slots = []
    for i in range(n):
        box = boxes[i] if i < len(boxes) else [0.1, 0.1, 0.9, 0.9]
        slots.append(
            HydraSlot(
                slot_id=f"{noun}_{i + 1}",
                prompt=f"distinct {noun} instance {i + 1}, unique pose, unique silhouette",
                box=list(box),
            )
        )
    layout = {
        "mode": "hydra_slots",
        "anti_bleed": True,
        "global_prompt": prompt,
        "regions": [{"id": s.slot_id, "prompt": s.prompt, "box": s.box} for s in slots],
    }
    return HydraSlotPlan(
        slots=slots,
        global_prompt=prompt,
        box_layout=layout,
        notes=[f"{n} hydra slots for {noun}"],
    )
