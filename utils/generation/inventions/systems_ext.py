"""Systems inventions 83–90 + self-healing loop #100."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.generation.inventions.failure_oracle import diagnose_failures
from utils.generation.inventions.stack import apply_invention_stack

__all__ = [
    "adaptive_nfe",
    "energy_reject",
    "ErrorRedoPlan",
    "plan_error_redo",
    "oracle_search_rungs",
    "SelfHealingPlan",
    "plan_self_healing",
    "run_self_healing_plan",
    "flow_vp_handoff_spec",
    "apex_distill_spec",
    "parallel_expert_merge_spec",
]


def adaptive_nfe(prompt: str, *, base_steps: int = 28, min_steps: int = 12, max_steps: int = 40) -> int:
    """Oracle risk → step budget (#86)."""
    rep = diagnose_failures(prompt)
    risk = len(rep.risks)
    steps = int(base_steps + 3 * risk - (2 if not rep.risks else 0))
    return int(max(min_steps, min(max_steps, steps)))


def energy_reject(sat_score: float, aesthetic: float = 0.5, *, threshold: float = 0.55) -> bool:
    """Reject sample when constraint energy too high (#90)."""
    energy = (1.0 - float(sat_score)) * 0.7 + (1.0 - float(aesthetic)) * 0.3
    return energy > (1.0 - threshold)


@dataclass
class ErrorRedoPlan:
    regions: list[str] = field(default_factory=list)
    max_redos: int = 2
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_error_redo(risks: list[str]) -> ErrorRedoPlan:
    regions = []
    for r in risks:
        if r in ("anatomy",):
            regions.extend(["hands", "face"])
        if r == "glyph":
            regions.append("subject")
        if r == "plastic":
            regions.append("face")
    return ErrorRedoPlan(regions=list(dict.fromkeys(regions)), notes=["mid-sample region redo when critic fires"])


def oracle_search_rungs(prompt: str) -> list[dict[str, Any]]:
    """Combine TTS rungs with oracle (#83)."""
    steps = adaptive_nfe(prompt)
    return [
        {"rung": 0, "steps": max(6, steps // 3), "pool": 4},
        {"rung": 1, "steps": max(10, steps // 2), "pool": 2},
        {"rung": 2, "steps": steps, "pool": 1},
    ]


@dataclass
class SelfHealingPlan:
    """#100 — full loop descriptor."""

    prompt: str
    negative: str = ""
    steps: int = 28
    scheduler: str = "apex"
    solver: str = "apex"
    invention_enable: str = "auto"
    max_repair_iters: int = 2
    sample_argv: list[str] = field(default_factory=list)
    critique_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_self_healing(prompt: str, negative: str = "") -> SelfHealingPlan:
    inv = apply_invention_stack(prompt, negative, enable="auto")
    steps = adaptive_nfe(inv.positive)
    redo = plan_error_redo((inv.reports.get("oracle") or {}).get("risks") or [])
    argv = [
        "--prompt",
        inv.positive,
        "--negative-prompt",
        inv.negative or negative,
        "--steps",
        str(steps),
        "--scheduler",
        "apex",
        "--solver",
        "apex_adaptive",
        "--invention-stack",
        "off",  # already applied
        "--invention-spectra",
    ]
    return SelfHealingPlan(
        prompt=inv.positive,
        negative=inv.negative,
        steps=steps,
        sample_argv=argv,
        critique_hints=redo.regions,
        notes=["oracle→stack→apex→critique→inpaint loop", f"risks={(inv.reports.get('oracle') or {}).get('risks')}"],
    )


def run_self_healing_plan(plan: SelfHealingPlan, *, work_dir: str | Path) -> Path:
    w = Path(work_dir)
    w.mkdir(parents=True, exist_ok=True)
    path = w / "self_healing_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return path


def flow_vp_handoff_spec() -> dict[str, Any]:
    return {"status": "moonshot", "idea": "early flow ODE then VP refinement near clean"}


def apex_distill_spec() -> dict[str, Any]:
    return {"status": "moonshot", "idea": "distill APEX adaptive teacher into few-step student"}


def parallel_expert_merge_spec() -> dict[str, Any]:
    return {"status": "moonshot", "idea": "structure expert + detail expert denoisers, merge by SPECTRA weights"}
