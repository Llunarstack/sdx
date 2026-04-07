"""Legacy compatibility namespace for `vit_quality`."""

from __future__ import annotations

import warnings

from vit_quality import *  # noqa: F403

warnings.warn(
    "`ViT` is a legacy compatibility namespace; prefer `vit_quality` for new imports.",
    DeprecationWarning,
    stacklevel=2,
)
