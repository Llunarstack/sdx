"""Compatibility shim — canonical module: `config.defaults.style_artists`."""

from __future__ import annotations

import warnings

from .defaults.style_artists import *  # noqa: F403

warnings.warn(
    "`config.style_artists` is a compatibility shim; import `config.defaults.style_artists`.",
    DeprecationWarning,
    stacklevel=2,
)
