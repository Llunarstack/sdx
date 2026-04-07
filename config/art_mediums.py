"""Compatibility shim — canonical module: `config.defaults.art_mediums`."""

from __future__ import annotations

import warnings

from .defaults.art_mediums import *  # noqa: F403

warnings.warn(
    "`config.art_mediums` is a compatibility shim; import `config.defaults.art_mediums`.",
    DeprecationWarning,
    stacklevel=2,
)

