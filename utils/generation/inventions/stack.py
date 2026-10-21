"""Apply the invention prompt passes (and an optional box layout) in one call."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.generation.inventions.bindlock import apply_bindlock_to_prompts, plan_bindlock
from utils.generation.inventions.countgate import apply_countgate_prompts, plan_countgate, plan_hydra_slots
from utils.generation.inventions.failure_oracle import diagnose_failures, repair_plan_from_failures
from utils.generation.inventions.negatron import apply_negatron
from utils.generation.inventions.priors import apply_anatomon_prompts, apply_friction_to_prompts, scene_priors

__all__ = ["InventionStackResult", "apply_invention_stack"]

# Oracle repair module -> stack pass (``ideogram`` has no prompt pass).
_ORACLE_PASS = {"edit_skills": "anatomon", "multi_char_cast": "hydra", "ideogram": ""}


@dataclass
class InventionStackResult:
    positive: str
    negative: str
    box_layout: dict[str, Any] | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    repairs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_invention_stack(
    prompt: str,
    negative: str = "",
    *,
    enable: str = "auto",
    work_dir: str | Path = "",
) -> InventionStackResult:
    """
    ``enable``:
      - ``off``: no-op
      - ``auto``: failure oracle picks passes, plus the scene priors that fit the prompt
      - ``all``: bindlock, countgate, negatron, friction, anatomon, hydra + scene priors
      - comma list: e.g. ``bindlock,negatron,friction`` (``texture`` / ``compose`` add the scene priors)
    """
    pos, neg = str(prompt or ""), str(negative or "")
    mode = str(enable or "auto").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return InventionStackResult(positive=pos, negative=neg)

    oracle = repair_plan_from_failures(diagnose_failures(pos, negative=neg))
    if mode == "all":
        wanted = {"bindlock", "countgate", "negatron", "friction", "anatomon", "hydra"}
    elif mode == "auto":
        wanted = {_ORACLE_PASS.get(m, m) for r in oracle.repairs if (m := str(r.get("module") or ""))} - {""}
        wanted = wanted or {"friction"}
    else:
        wanted = {x.strip() for x in mode.split(",") if x.strip()}

    reports: dict[str, Any] = {"oracle": oracle.to_dict()}
    box_layout = None

    if "negatron" in wanted:
        pos, neg, plan = apply_negatron(pos, neg)
        reports["negatron"] = plan.to_dict()

    if "bindlock" in wanted:
        plan = plan_bindlock(pos)
        pos, neg = apply_bindlock_to_prompts(pos, neg, plan)
        reports["bindlock"] = plan.to_dict()

    if "countgate" in wanted:
        plan = plan_countgate(pos)
        pos, neg = apply_countgate_prompts(pos, neg, plan)
        reports["countgate"] = plan.to_dict()
        if plan.boxes:
            box_layout = {
                "mode": "countgate",
                "anti_bleed": True,
                "global_prompt": pos,
                "regions": [
                    {"id": f"inst_{i}", "prompt": f"instance {i + 1}", "box": b} for i, b in enumerate(plan.boxes)
                ],
            }

    if "hydra" in wanted:
        hp = plan_hydra_slots(pos)
        reports["hydra"] = hp.to_dict()
        if hp.box_layout and hp.slots:
            box_layout = hp.box_layout

    if "anatomon" in wanted:
        pos, neg, plan = apply_anatomon_prompts(pos, neg)
        reports["anatomon"] = plan.to_dict()

    if "friction" in wanted:
        pos, neg = apply_friction_to_prompts(pos, neg)
        reports["friction"] = {"applied": True}

    if mode in ("auto", "all") or "texture" in wanted or "compose" in wanted:
        priors = scene_priors(pos)
        for ap, an in priors:
            if ap and ap.lower() not in pos.lower():
                pos = f"{pos}, {ap}"
            if an:
                neg = f"{neg}, {an}".strip(", ")
        reports["priors"] = [ap for ap, _ in priors]

    result = InventionStackResult(
        positive=pos,
        negative=neg,
        box_layout=box_layout,
        reports=reports,
        repairs=list(oracle.repairs),
    )
    if work_dir:
        w = Path(work_dir)
        w.mkdir(parents=True, exist_ok=True)
        (w / "invention_stack.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        if box_layout:
            (w / "invention_box_layout.json").write_text(json.dumps(box_layout, indent=2), encoding="utf-8")
    return result
