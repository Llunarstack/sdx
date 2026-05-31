"""
**Trajectory experience** distillation (GenEvolve-style best–worst comparison).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .state import TrajectoryRecord


@dataclass(slots=True)
class TrajectoryExperience:
    """Structured lesson from comparing trajectories on the same goal."""

    goal_prompt: str
    best_id: str
    worst_id: str
    prompt_delta: str
    negative_delta: str
    metric_delta: dict[str, float] = field(default_factory=dict)
    tool_delta: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def distill_trajectory_experience(
    goal_prompt: str,
    trajectories: Sequence[TrajectoryRecord],
) -> Optional[TrajectoryExperience]:
    """
    Compare best vs worst trajectory; abstract prompt/tool differences.
    """
    if len(trajectories) < 2:
        return None
    ranked = sorted(trajectories, key=lambda t: float(t.composite), reverse=True)
    best, worst = ranked[0], ranked[-1]
    if best.trajectory_id == worst.trajectory_id:
        return None

    def _unique_suffix(a: str, b: str) -> str:
        al = set(x.strip().lower() for x in a.split(",") if x.strip())
        bl = set(x.strip().lower() for x in b.split(",") if x.strip())
        extra = [x for x in al - bl if x]
        return ", ".join(sorted(extra)[:8])

    prompt_delta = _unique_suffix(best.prompt_final, worst.prompt_final)
    neg_delta = _unique_suffix(worst.negative_prompt, best.negative_prompt)
    tool_delta = [t for t in best.tool_sequence if t not in worst.tool_sequence]
    metric_delta = {
        k: float(getattr(best, "metrics", {}).get(k, 0.0)) - float(getattr(worst, "metrics", {}).get(k, 0.0))
        for k in set(list(getattr(best, "metrics", {}).keys()) + list(getattr(worst, "metrics", {}).keys()))
    }
    return TrajectoryExperience(
        goal_prompt=goal_prompt,
        best_id=best.trajectory_id,
        worst_id=worst.trajectory_id,
        prompt_delta=prompt_delta,
        negative_delta=neg_delta,
        metric_delta=metric_delta,
        tool_delta=tool_delta,
    )


def append_experience_memory(path: Path, exp: TrajectoryExperience) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(exp.to_json() + "\n")


def load_experience_memory(path: Path, *, limit: int = 32) -> List[TrajectoryExperience]:
    if not path.is_file():
        return []
    out: List[TrajectoryExperience] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(
                TrajectoryExperience(
                    goal_prompt=str(d.get("goal_prompt", "")),
                    best_id=str(d.get("best_id", "")),
                    worst_id=str(d.get("worst_id", "")),
                    prompt_delta=str(d.get("prompt_delta", "")),
                    negative_delta=str(d.get("negative_delta", "")),
                    metric_delta=dict(d.get("metric_delta", {})),
                    tool_delta=list(d.get("tool_delta", [])),
                )
            )
        except json.JSONDecodeError:
            continue
    return out[-limit:]


def apply_memory_to_prompt(prompt: str, memory: Sequence[TrajectoryExperience]) -> str:
    """Merge recurring prompt deltas from experience memory."""
    if not memory:
        return prompt
    bits: list[str] = []
    seen: set[str] = set()
    pl = prompt.lower()
    for exp in memory:
        for chunk in exp.prompt_delta.split(","):
            c = chunk.strip()
            if c and c.lower() not in seen and c.lower() not in pl:
                seen.add(c.lower())
                bits.append(c)
    if not bits:
        return prompt
    return prompt + ", " + ", ".join(bits[:6])


__all__ = [
    "TrajectoryExperience",
    "append_experience_memory",
    "apply_memory_to_prompt",
    "distill_trajectory_experience",
    "load_experience_memory",
]
