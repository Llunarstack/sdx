"""Compatibility shim — canonical module: `config.defaults.ai_image_shortcomings`."""

from __future__ import annotations

import warnings

from .defaults.ai_image_shortcomings import *  # noqa: F403

warnings.warn(
    "`config.ai_image_shortcomings` is a compatibility shim; "
    "import `config.defaults.ai_image_shortcomings`.",
    DeprecationWarning,
    stacklevel=2,
)
