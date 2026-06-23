"""Two-stage cascaded diffusion: a base model followed by a refiner.

Cascaded generation splits the work: a base model produces coarse latents, then a
second refiner model sharpens them. This wrapper just chains the two and returns
both outputs (the base latents are kept so they can be supervised too).

The optional **RAE latent bridge** handles a mismatch in latent spaces: if the
outer pipeline works in an RAE latent with a non-standard channel count, the bridge
maps it into the 4-channel space the DiT expects on the way in and back out again.
Its cycle-consistency loss (RAE -> DiT -> RAE should be identity) is surfaced so the
trainer can keep that mapping faithful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from models.rae_latent_bridge import RAELatentBridge


@dataclass(slots=True)
class CascadedPipelineOutput:
    """Outputs of one cascaded forward pass.

    ``base_latents`` is the stage-1 result, ``refined_latents`` the stage-2 result
    (mapped back to the input latent space), and ``bridge_cycle_loss`` is the RAE
    bridge's cycle-consistency loss when the bridge is active, else ``None``.
    """

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
        """Map incoming latents into DiT space (no-op without a bridge); also return its cycle loss."""
        if self.bridge is None:
            return latents, None
        z_dit = self.bridge.rae_to_dit(latents)
        cycle = self.bridge.cycle_loss(latents)
        return z_dit, cycle

    def _adapt_out(self, latents: torch.Tensor) -> torch.Tensor:
        """Map DiT-space latents back to the outer latent space (no-op without a bridge)."""
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
