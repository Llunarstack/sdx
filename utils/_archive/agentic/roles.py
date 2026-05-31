"""
**Multi-role pipeline** — Designer / Verifier / Reasoner as explicit agent stages.

Maps industry multi-expert patterns onto ``ToolRegistry`` skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from utils.generation.orchestration import DESIGNER, REASONER, VERIFIER

from .state import AgentContext
from .tools import AgentTool, ToolRegistry


@dataclass(slots=True)
class RoleStageResult:
    role: str
    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolePipelineResult:
    stages: List[RoleStageResult] = field(default_factory=list)
    out_path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    prompt_final: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.stages) and all(s.ok for s in self.stages)


class RolePipeline:
    """
    Run Reasoner → Designer → Verifier in order (no reflect loop).

    Use ``ImageGenerationAgent`` when you need Act–Reflect–Think–Act retries.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        self.registry = registry

    @classmethod
    def from_context(
        cls,
        ctx: AgentContext,
        *,
        repo_root: Path,
        **registry_kw: Any,
    ) -> RolePipeline:
        reg = ToolRegistry(ctx, repo_root=repo_root, **registry_kw)
        return cls(reg)

    def run_reasoner(self) -> RoleStageResult:
        r1 = self.registry.execute(AgentTool.rag_retrieve)
        r2 = self.registry.execute(AgentTool.expand_prompt)
        ok = r1.ok and r2.ok
        return RoleStageResult(
            role=REASONER.name,
            ok=ok,
            message=f"rag={r1.message} expand={r2.message}",
            data={"prompt": self.registry.session["prompt"], "facts": r1.data.get("facts", [])},
        )

    def run_designer(self, *, out_suffix: str = "") -> RoleStageResult:
        r = self.registry.execute(AgentTool.generate, out_suffix=out_suffix)
        return RoleStageResult(
            role=DESIGNER.name,
            ok=r.ok,
            message=r.message,
            data={"out": r.data.get("out", ""), "cmd": r.data.get("cmd", [])},
        )

    def run_verifier(self) -> RoleStageResult:
        r = self.registry.execute(AgentTool.verify)
        return RoleStageResult(
            role=VERIFIER.name,
            ok=r.ok,
            message=r.message,
            data=dict(r.data),
        )

    def run(self, *, out_suffix: str = "") -> RolePipelineResult:
        stages = [
            self.run_reasoner(),
            self.run_designer(out_suffix=out_suffix),
            self.run_verifier(),
        ]
        return RolePipelineResult(
            stages=stages,
            out_path=str(self.registry.session.get("last_out", "")),
            metrics=dict(self.registry.session.get("metrics", {})),
            prompt_final=str(self.registry.session.get("prompt", "")),
        )


__all__ = ["RolePipeline", "RolePipelineResult", "RoleStageResult"]
