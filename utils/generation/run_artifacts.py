"""
Canonical filenames for training run reproducibility artifacts.

``train.py`` writes these next to checkpoints; docs and tools can import the
constants instead of duplicating string literals.
"""

from __future__ import annotations

from pathlib import Path

RUN_MANIFEST_FILENAME = "run_manifest.json"
TRAIN_CONFIG_SNAPSHOT_FILENAME = "config.train.json"


def train_artifact_paths(exp_dir: Path) -> tuple[Path, Path]:
    """Return ``(config.train.json, run_manifest.json)`` paths under ``exp_dir``."""
    d = Path(exp_dir)
    return d / TRAIN_CONFIG_SNAPSHOT_FILENAME, d / RUN_MANIFEST_FILENAME
