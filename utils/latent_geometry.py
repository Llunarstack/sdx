"""
Re-export for backward compatibility.

Implementation: :mod:`sdx_native.latent_geometry` under ``native/python/sdx_native/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_np = Path(__file__).resolve().parent.parent / "native" / "python"
if str(_np) not in sys.path:
    sys.path.insert(0, str(_np))

from sdx_native.latent_geometry import *  # noqa: F403
