"""
Foreign VAE **probe + latent bridge** — can this autoencoder feed the sdx DiT?

A VAE is swappable only if its latent tensor matches what the DiT was trained
on (channel count, roughly compatible scaling). The probe reads those facts
straight from the state dict; :class:`LatentBridge` is the adapter for the
mismatch case — identity-initialized when channels agree (safe to insert
anywhere), a 1×1 conv otherwise (needs a short calibration fine-tune before
outputs are meaningful).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from utils.compat.asset_sniffer import load_state_dict

__all__ = [
    "LatentBridge",
    "VAEProbe",
    "probe_vae",
]

_FAMILY_BY_CHANNELS: dict[int, tuple[str, float]] = {
    4: ("sd (kl-f8)", 0.18215),
    16: ("flux/sd3 (16ch)", 0.3611),
    32: ("high-channel research vae", 1.0),
}


@dataclass(slots=True)
class VAEProbe:
    """Facts about a foreign VAE relevant to swapping it in."""

    latent_channels: int = 0
    family_guess: str = "unknown"
    scaling_factor_guess: float = 1.0
    has_encoder: bool = False
    has_decoder: bool = False


def probe_vae(path_or_state: str | Path | dict) -> VAEProbe:
    """Read latent geometry from a VAE state dict (diffusers or CompVis layout)."""
    state = load_state_dict(path_or_state)
    probe = VAEProbe()
    prefixes = ("", "first_stage_model.")
    for pre in prefixes:
        w = state.get(f"{pre}decoder.conv_in.weight")
        if w is not None and w.ndim == 4:
            probe.latent_channels = int(w.shape[1])
            probe.has_decoder = True
            break
    for pre in prefixes:
        if any(k.startswith(f"{pre}encoder.") for k in state):
            probe.has_encoder = True
            break
    if probe.latent_channels:
        fam, sf = _FAMILY_BY_CHANNELS.get(probe.latent_channels, ("unknown", 1.0))
        probe.family_guess = fam
        probe.scaling_factor_guess = sf
    return probe


class LatentBridge(nn.Module):
    """
    1×1 conv adapter between a foreign VAE's latent space and the DiT's.

    Same channel count → zero-init residual (identity at load, refinable).
    Different count → plain projection that must be calibrated (train briefly
    against paired encodes) before generation quality is usable.
    """

    def __init__(self, source_channels: int, target_channels: int):
        super().__init__()
        self.identity_start = int(source_channels) == int(target_channels)
        self.proj = nn.Conv2d(int(source_channels), int(target_channels), kernel_size=1)
        if self.identity_start:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.proj(latent) if self.identity_start else self.proj(latent)
