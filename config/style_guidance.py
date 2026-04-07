"""Compatibility shim — canonical module: `config.defaults.style_guidance`."""

from __future__ import annotations

import warnings

from .defaults.style_guidance import *  # noqa: F403

warnings.warn(
    "`config.style_guidance` is a compatibility shim; import `config.defaults.style_guidance`.",
    DeprecationWarning,
    stacklevel=2,
)

