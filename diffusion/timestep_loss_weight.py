"""Compatibility shim — canonical module: `diffusion.losses.timestep_loss_weight`."""

from __future__ import annotations

import warnings

from .losses.timestep_loss_weight import *  # noqa: F403

warnings.warn(
    "`diffusion.timestep_loss_weight` is a compatibility shim; "
    "import `diffusion.losses.timestep_loss_weight`.",
    DeprecationWarning,
    stacklevel=2,
)
