from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .rae_latent_bridge import RAELatentBridge


@dataclass
class CascadedSchedule:
    base_steps: int = 30
    refine_steps: int = 20
    guidance_scale: float = 5.0


class CascadedMultimodalDiffusion(nn.Module):
    """
    Scaffolding module for:
      Native multimodal conditioning + cascaded diffusion + RAE latent bridge.

    - base_model: coarse generation stage
    - refine_model: detail/refinement stage
    - bridge: optional RAE<->DiT channel adapter

    Optional **blend_base_refine** mixes base and refine outputs (ablation / stability).
    **detach_base_for_refine** stops gradients into the base model through the refine path
    when refine is trained with a frozen base.
    """

    def __init__(
        self,
        base_model: nn.Module,
        refine_model: nn.Module,
        *,
        bridge: Optional[RAELatentBridge] = None,
        blend_base_refine: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.refine_model = refine_model
        self.bridge = bridge
        if not (0.0 <= float(blend_base_refine) <= 1.0):
            raise ValueError("blend_base_refine must be in [0, 1]")
        self.blend_base_refine = float(blend_base_refine)

    def _to_dit_latents(self, z: torch.Tensor) -> torch.Tensor:
        if self.bridge is None:
            return z
        if z.shape[1] == self.bridge.rae_channels:
            return self.bridge.rae_to_dit(z)
        return z

    def _to_rae_latents(self, z: torch.Tensor) -> torch.Tensor:
        if self.bridge is None:
            return z
        if z.shape[1] == self.bridge.dit_channels:
            return self.bridge.dit_to_rae(z)
        return z

    def _blend(self, base_out: torch.Tensor, refine_out: torch.Tensor) -> torch.Tensor:
        w = self.blend_base_refine
        return (1.0 - w) * base_out + w * refine_out

    def forward(
        self,
        x: torch.Tensor,
        t_base: torch.Tensor,
        t_refine: torch.Tensor,
        *,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        output_rae_latents: bool = False,
        detach_base_for_refine: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            detach_base_for_refine: if True, ``refine_model`` sees ``base_out`` detached (no grad to base).
        """
        kw = model_kwargs or {}
        x_dit = self._to_dit_latents(x)

        base_out = self.base_model(x_dit, t_base, encoder_hidden_states=encoder_hidden_states, **kw)
        base_for_refine = base_out.detach() if detach_base_for_refine else base_out
        refine_out = self.refine_model(base_for_refine, t_refine, encoder_hidden_states=encoder_hidden_states, **kw)

        blended = self._blend(base_out, refine_out)
        out = {
            "base_output": base_out,
            "refine_output": refine_out,
            "final_output": blended,
        }
        if output_rae_latents:
            out["final_output"] = self._to_rae_latents(blended)
        return out

    def bridge_cycle_loss(self, z_rae: torch.Tensor) -> torch.Tensor:
        if self.bridge is None:
            return z_rae.new_zeros(())
        return self.bridge.cycle_loss(z_rae)
