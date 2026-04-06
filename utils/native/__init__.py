"""
Unified native bridge for backward-compatible imports.

This module replaces multiple thin wrapper files under ``utils/native/`` by:
- ensuring ``native/python`` is importable, and
- re-exporting selected symbols from ``sdx_native`` modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

_NP = Path(__file__).resolve().parents[2] / "native" / "python"
if str(_NP) not in sys.path:
    sys.path.insert(0, str(_NP))

from sdx_native.cuda_image_metrics_native import *  # noqa: E402,F403
from sdx_native.image_metrics_native import *  # noqa: E402,F403
from sdx_native.latent_geometry import *  # noqa: E402,F403
from sdx_native.native_tools import *  # noqa: E402,F403
from sdx_native.score_ops_native import *  # noqa: E402,F403
from sdx_native.text_hygiene import *  # noqa: E402,F403
