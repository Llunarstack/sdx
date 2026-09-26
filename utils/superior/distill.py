"""Training-side distillation helpers (LADD + step planning)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.training.ladd_distillation import LADDConfig


@dataclass(slots=True)
class DistillStepPlan:
    """Suggested schedule for teacher→student distillation runs."""

    warmup_steps: int = 500
    teacher_freeze: bool = True
    student_lr: float = 1e-5
    discriminator_lr: float = 2e-5
    log_every: int = 50
    save_every: int = 1000


def ladd_config_from_train(cfg: Any) -> LADDConfig:
    """Build ``LADDConfig`` from ``TrainConfig``-like object (optional attrs)."""
    return LADDConfig(
        mse_teacher=float(getattr(cfg, "ladd_mse_teacher", 1.0) or 1.0),
        adversarial=float(getattr(cfg, "ladd_adversarial", 0.1) or 0.1),
        r1_gamma=float(getattr(cfg, "ladd_r1_gamma", 0.0) or 0.0),
    )


__all__ = ["DistillStepPlan", "LADDConfig", "ladd_config_from_train"]
