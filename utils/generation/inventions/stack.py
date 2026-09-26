"""Apply Invention Lab stack to prompts (and optional box layout) in one call."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["InventionStackResult", "apply_invention_stack"]


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
    enable: str = "auto",  # auto|all|off|comma list
    work_dir: str | Path = "",
) -> InventionStackResult:
    """
    ``enable``:
      - ``off``: no-op
      - ``auto``: failure oracle picks modules
      - ``all``: run bindlock, countgate, negatron, friction, anatomon, hydra
      - comma list: e.g. ``bindlock,negatron,friction``
    """
    pos, neg = str(prompt or ""), str(negative or "")
    mode = str(enable or "auto").strip().lower()
    if mode in ("off", "none", "0", "false"):
        return InventionStackResult(positive=pos, negative=neg)

    from utils.generation.inventions.failure_oracle import diagnose_failures, repair_plan_from_failures

    oracle = repair_plan_from_failures(diagnose_failures(pos, negative=neg))
    wanted: set[str]
    if mode == "all":
        wanted = {"bindlock", "countgate", "negatron", "friction", "anatomon", "hydra", "spectra"}
    elif mode == "auto":
        wanted = set()
        for r in oracle.repairs:
            m = str(r.get("module") or "")
            if m == "ideogram":
                continue
            if m == "edit_skills":
                wanted.add("anatomon")
            elif m == "multi_char_cast":
                wanted.add("hydra")
            elif m:
                wanted.add(m)
        if not wanted:
            wanted = {"friction"}  # mild default quality
    else:
        wanted = {x.strip() for x in mode.split(",") if x.strip()}

    reports: dict[str, Any] = {"oracle": oracle.to_dict()}
    box_layout = None

    if "negatron" in wanted:
        from utils.generation.inventions.negatron import apply_negatron

        pos, neg, plan = apply_negatron(pos, neg)
        reports["negatron"] = plan.to_dict()

    if "bindlock" in wanted:
        from utils.generation.inventions.bindlock import apply_bindlock_to_prompts, plan_bindlock

        plan = plan_bindlock(pos)
        pos, neg = apply_bindlock_to_prompts(pos, neg, plan)
        reports["bindlock"] = plan.to_dict()

    if "countgate" in wanted:
        from utils.generation.inventions.countgate import apply_countgate_prompts, plan_countgate

        plan = plan_countgate(pos)
        pos, neg = apply_countgate_prompts(pos, neg, plan)
        reports["countgate"] = plan.to_dict()
        if plan.boxes and box_layout is None:
            box_layout = {
                "mode": "countgate",
                "anti_bleed": True,
                "global_prompt": pos,
                "regions": [
                    {"id": f"inst_{i}", "prompt": f"instance {i + 1}", "box": b} for i, b in enumerate(plan.boxes)
                ],
            }

    if "hydra" in wanted:
        from utils.generation.inventions.hydra_slots import plan_hydra_slots

        hp = plan_hydra_slots(pos)
        reports["hydra"] = hp.to_dict()
        if hp.box_layout and hp.slots:
            box_layout = hp.box_layout

    if "anatomon" in wanted:
        from utils.generation.inventions.anatomon import apply_anatomon_prompts

        pos, neg, plan = apply_anatomon_prompts(pos, neg)
        reports["anatomon"] = plan.to_dict()

    if "friction" in wanted:
        from utils.generation.inventions.friction_texture import apply_friction_to_prompts

        pos, neg = apply_friction_to_prompts(pos, neg)
        reports["friction"] = {"applied": True}

    # Extended priors (composition + anatomy + texture) when auto/all
    if mode in ("auto", "all") or "texture" in wanted or "compose" in wanted:
        from utils.generation.inventions.anatomy_ext import (
            contact_shadow_addon,
            eye_symmetry_breaker_addon,
            plan_hand_expert,
            teeth_mouth_addon,
        )
        from utils.generation.inventions.composition_ext import anti_centering_addon, zorder_addon
        from utils.generation.inventions.texture_ext import (
            anti_beauty_filter_addon,
            filmic_highlight_addon,
            plan_lens_exif,
        )

        extras = [
            anti_centering_addon(),
            zorder_addon(pos),
            eye_symmetry_breaker_addon(),
            teeth_mouth_addon(),
            contact_shadow_addon(),
            anti_beauty_filter_addon(),
            filmic_highlight_addon(),
        ]
        for ap, an in extras:
            if ap and ap.lower() not in pos.lower():
                pos = f"{pos}, {ap}"
            if an:
                neg = f"{neg}, {an}".strip(", ")
        he = plan_hand_expert(pos)
        if he.active:
            pos = f"{pos}, {he.positive}"
            neg = f"{neg}, {he.negative}".strip(", ")
            reports["hand_expert"] = he.to_dict()
        lens = plan_lens_exif()
        if lens.positive.lower() not in pos.lower():
            pos = f"{pos}, {lens.positive}"
        reports["lens_exif"] = lens.to_dict()

    if "spectra" in wanted:
        reports["spectra"] = {"hint": "use --invention-spectra for CFG/HF schedule in loop"}

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
