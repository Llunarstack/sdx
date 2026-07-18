"""Shared solver state for multistep ODE samplers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SolverState:
    """History buffer for multistep solvers (DPM++ / UniPC / flow AB)."""

    model_outputs: list[torch.Tensor] = field(default_factory=list)
    timesteps: list[float] = field(default_factory=list)  # λ (VP) or s (flow)
    max_order: int = 3

    def push(self, model_output: torch.Tensor, time_value: float) -> None:
        self.model_outputs.append(model_output)
        self.timesteps.append(float(time_value))
        while len(self.model_outputs) > int(self.max_order):
            self.model_outputs.pop(0)
            self.timesteps.pop(0)

    def clear(self) -> None:
        self.model_outputs.clear()
        self.timesteps.clear()

    @property
    def order(self) -> int:
        return len(self.model_outputs)
