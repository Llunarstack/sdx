"""Validator rules for flow matching vs MDM."""

from pathlib import Path

from config.train_config import TrainConfig
from utils.training.config_validator import validate_train_config


def test_flow_matching_excludes_mdm(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    cfg = TrainConfig(
        data_path=str(root),
        flow_matching_training=True,
        mdm_mask_ratio=0.25,
    )
    issues = validate_train_config(cfg, require_cuda=False)
    assert any("flow_matching_training" in msg for msg in issues)
