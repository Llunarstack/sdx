from __future__ import annotations

import torch


class ModelEMA:
    """Simple EMA for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        d = self.decay
        for k, v in msd.items():
            if k not in self.shadow or not v.dtype.is_floating_point:
                continue
            self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            if k in self.shadow:
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}
