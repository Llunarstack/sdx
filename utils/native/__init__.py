"""
Unified native bridge for backward-compatible imports.

This module replaces multiple thin wrapper files under ``utils/native/`` by:
- ensuring ``native/python`` is importable, and
- re-exporting selected symbols from ``sdx_native`` modules.

Also re-exports :attr:`sdx_native.__all__` (NumPy fast paths, caption CSV helpers,
optional ``c_buffer_stats`` ctypes, diffusion sigma/SNR numpy, etc.) so
``from utils.native import normalize_caption_csv, scan_file_chunks`` works.
"""

from __future__ import annotations

import sys
from pathlib import Path

_NP = Path(__file__).resolve().parents[2] / "native" / "python"
if str(_NP) not in sys.path:
    sys.path.insert(0, str(_NP))

from sdx_native import *  # noqa: E402,F403,F405
from sdx_native.cuda_image_metrics_native import *  # noqa: E402,F403
from sdx_native.image_metrics_native import *  # noqa: E402,F403
from sdx_native.latent_geometry import *  # noqa: E402,F403
from sdx_native.native_tools import *  # noqa: E402,F403
from sdx_native.score_ops_native import *  # noqa: E402,F403
from sdx_native.text_hygiene import *  # noqa: E402,F403
