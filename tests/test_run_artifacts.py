from __future__ import annotations

from pathlib import Path

from utils.generation.run_artifacts import (
    RUN_MANIFEST_FILENAME,
    TRAIN_CONFIG_SNAPSHOT_FILENAME,
    train_artifact_paths,
)


def test_train_artifact_paths() -> None:
    cfg_p, man_p = train_artifact_paths(Path("results/exp1"))
    assert cfg_p == Path("results/exp1") / TRAIN_CONFIG_SNAPSHOT_FILENAME
    assert man_p == Path("results/exp1") / RUN_MANIFEST_FILENAME
    assert cfg_p.name == "config.train.json"
    assert man_p.name == "run_manifest.json"
