"""
Step-based **training timestep distribution** (curriculum over ``global_step``).

When ``TrainConfig.timestep_curriculum_schedule`` is non-empty, it overrides the flat
``timestep_sample_mode`` / logit parameters for the active phase. Phases are
``start_step:mode`` segments ordered by ``start_step``; the last phase with
``start_step <= global_step`` wins.

Schedule syntax (segments separated by ``|``)::

    0:high_noise|50000:logit_normal:0:1|120000:low_noise

Each segment::

    <start_step>:<mode>[:<logit_mean>:<logit_std>]

Optional ``logit_mean`` / ``logit_std`` apply to ``logit_normal``; if omitted, fall back
to ``TrainConfig.timestep_logit_mean`` / ``timestep_logit_std``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_CACHE: dict[str, list[TimestepCurriculumPhase]] = {}


@dataclass(slots=True, frozen=True)
class TimestepCurriculumPhase:
    start_step: int
    mode: str
    logit_mean: Optional[float] = None
    logit_std: Optional[float] = None


def parse_timestep_curriculum_schedule(schedule: str) -> list[TimestepCurriculumPhase]:
    """Parse a non-empty curriculum string; raises ``ValueError`` on bad input."""
    text = str(schedule or "").strip()
    if not text:
        return []
    if text in _CACHE:
        return list(_CACHE[text])

    phases: list[TimestepCurriculumPhase] = []
    for raw_seg in text.split("|"):
        seg = raw_seg.strip()
        if not seg:
            continue
        parts = [p.strip() for p in seg.split(":")]
        if len(parts) < 2:
            raise ValueError(f"curriculum segment must be start:mode[:mean:std], got {seg!r}")
        start_step = int(parts[0])
        mode = parts[1].strip().lower().replace("-", "_")
        lm: Optional[float] = None
        ls: Optional[float] = None
        if len(parts) >= 4:
            lm = float(parts[2])
            ls = float(parts[3])
        elif len(parts) == 3:
            raise ValueError(f"curriculum segment {seg!r}: provide both mean and std after mode, or neither")
        phases.append(TimestepCurriculumPhase(start_step=start_step, mode=mode, logit_mean=lm, logit_std=ls))

    if not phases:
        return []
    phases.sort(key=lambda p: p.start_step)
    _CACHE[text] = phases
    return list(phases)


def resolve_timestep_kwargs_for_step(cfg: Any, global_step: int) -> dict[str, Any]:
    """
    Resolve kwargs for ``diffusion.timestep_sampling.sample_training_timesteps``.

    When ``timestep_curriculum_schedule`` is set, picks the active phase by ``global_step``;
    otherwise uses ``timestep_sample_mode`` / logit fields on ``cfg``.
    """
    schedule = str(getattr(cfg, "timestep_curriculum_schedule", "") or "").strip()
    base_mode = str(getattr(cfg, "timestep_sample_mode", "uniform"))
    base_mean = float(getattr(cfg, "timestep_logit_mean", 0.0))
    base_std = float(getattr(cfg, "timestep_logit_std", 1.0))

    if not schedule:
        return {"mode": base_mode, "logit_mean": base_mean, "logit_std": base_std}

    phases = parse_timestep_curriculum_schedule(schedule)
    if not phases:
        return {"mode": base_mode, "logit_mean": base_mean, "logit_std": base_std}

    step = int(global_step)
    active = phases[0]
    for p in phases:
        if p.start_step <= step:
            active = p
        else:
            break

    mean = active.logit_mean if active.logit_mean is not None else base_mean
    std = active.logit_std if active.logit_std is not None else base_std
    return {"mode": active.mode, "logit_mean": float(mean), "logit_std": float(std)}
