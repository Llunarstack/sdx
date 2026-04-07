"""Compatibility shim — canonical module: `diffusion.losses.loss_weighting`."""

from __future__ import annotations

import warnings

from .losses.loss_weighting import *  # noqa: F403

warnings.warn(
    "`diffusion.loss_weighting` is a compatibility shim; import `diffusion.losses.loss_weighting`.",
    DeprecationWarning,
    stacklevel=2,
)
