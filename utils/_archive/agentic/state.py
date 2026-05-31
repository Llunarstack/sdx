"""
Agent **state**, traces, and trajectory records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class AgentContext:
    """Runtime context for one agent session."""

    ckpt: str
    prompt: str
    work_dir: str = "agentic_run"
    out: str = "agent_out.png"
    device: str = "cuda"
    negative_prompt: str = ""
    local_rag_jsonl: str = ""
    vit_ckpt: str = ""
    expected_text: str = ""
    qwen_path: str = ""
    reference_images: List[str] = field(default_factory=list)
    web_search: bool = True
    repo_root: Optional[str] = None
    dry_run: bool = False


@dataclass(slots=True)
class TrajectoryRecord:
    """One full tool-orchestrated attempt (GenEvolve-style trajectory)."""

    trajectory_id: str
    prompt_final: str
    negative_prompt: str
    tool_sequence: List[str]
    out_path: str
    composite: float = 0.0
    clip_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    notes: str = ""


@dataclass(slots=True)
class AgentTrace:
    """Full session trace for logs / experience distillation."""

    goal_prompt: str
    iterations: int = 0
    trajectories: List[TrajectoryRecord] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    experience_patches: List[str] = field(default_factory=list)
    best_trajectory_id: str = ""
    final_out: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_prompt": self.goal_prompt,
            "iterations": self.iterations,
            "trajectories": [asdict(t) for t in self.trajectories],
            "reflections": list(self.reflections),
            "experience_patches": list(self.experience_patches),
            "best_trajectory_id": self.best_trajectory_id,
            "final_out": self.final_out,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["AgentContext", "AgentTrace", "TrajectoryRecord"]
