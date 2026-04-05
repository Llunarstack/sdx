from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from models.rae_latent_bridge import RAELatentBridge


@dataclass
class CascadedPipelineOutput:
    base_latents: torch.Tensor
    refined_latents: torch.Tensor
    bridge_cycle_loss: Optional[torch.Tensor]


class CascadedMultimodalPipeline(nn.Module):
    """
    Minimal cascaded diffusion wrapper:
      stage-1 base model -> stage-2 refiner model
    with optional RAE latent bridge for non-4ch latent spaces.
    """

    def __init__(
        self,
        base_model: nn.Module,
        refiner_model: nn.Module,
        *,
        rae_channels: int = 0,
        dit_channels: int = 4,
    ):
        super().__init__()
        self.base_model = base_model
        self.refiner_model = refiner_model
        self.bridge = RAELatentBridge(rae_channels, dit_channels=dit_channels) if int(rae_channels) > 0 else None

    def _adapt_in(self, latents: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.bridge is None:
            return latents, None
        z_dit = self.bridge.rae_to_dit(latents)
        cycle = self.bridge.cycle_loss(latents)
        return z_dit, cycle

    def _adapt_out(self, latents: torch.Tensor) -> torch.Tensor:
        if self.bridge is None:
            return latents
        return self.bridge.dit_to_rae(latents)

    def forward(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> CascadedPipelineOutput:
        z_in, cycle_loss = self._adapt_in(latents)
        base = self.base_model(z_in, t, encoder_hidden_states=encoder_hidden_states, **kwargs)
        refined = self.refiner_model(base, t, encoder_hidden_states=encoder_hidden_states, **kwargs)
        refined_out = self._adapt_out(refined)
        return CascadedPipelineOutput(
            base_latents=base,
            refined_latents=refined_out,
            bridge_cycle_loss=cycle_loss,
        )
