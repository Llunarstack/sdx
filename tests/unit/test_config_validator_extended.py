"""Extra TrainConfig validation rules and inference arg semantics."""

import json
from pathlib import Path

from config.train_config import TrainConfig
from utils.training.config_validator import validate_inference_args, validate_train_config


def test_validate_train_config_image_size_multiple_of_8(tmp_path):
    cfg = TrainConfig(data_path=str(tmp_path), image_size=250)
    issues = validate_train_config(cfg, require_cuda=False)
    assert any("image_size" in x and "ERROR" in x for x in issues)


def test_validate_train_config_prediction_type(tmp_path):
    cfg = TrainConfig(data_path=str(tmp_path), prediction_type="bogus")
    issues = validate_train_config(cfg, require_cuda=False)
    assert any("prediction_type" in x for x in issues)


def test_validate_train_config_resolution_buckets_bad_pair(tmp_path):
    cfg = TrainConfig(data_path=str(tmp_path), resolution_buckets=[(256, 255)])
    issues = validate_train_config(cfg, require_cuda=False)
    assert any("resolution_buckets" in x for x in issues)


def test_validate_train_config_cuda_optional_no_error_on_cpu(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("{}\n", encoding="utf-8")
    cfg = TrainConfig(data_path=str(tmp_path), manifest_jsonl=str(m))
    issues = validate_train_config(cfg, require_cuda=False)
    assert not any(x.startswith("ERROR:") and "CUDA" in x for x in issues)
    import torch

    if not torch.cuda.is_available():
        assert any("CUDA" in x for x in issues)


def test_validate_inference_args_native_resolution_ok():
    class A:
        ckpt = ""
        width = 0
        height = 0
        steps = 50
        cfg_scale = 7.5

    issues = validate_inference_args(A())
    assert not any("ERROR" in x for x in issues)


def test_validate_inference_args_partial_dims_warning():
    class A:
        ckpt = ""
        width = 512
        height = 0
        steps = 50
        cfg_scale = 7.5

    issues = validate_inference_args(A())
    assert any("WARNING" in x for x in issues)


def test_validate_config_json_script_smoke(tmp_path):
    """Round-trip minimal JSON through the same constructor the dev script uses."""
    cfg = TrainConfig(data_path=str(tmp_path), manifest_jsonl=None)
    p = tmp_path / "cfg.json"
    # TrainConfig is a dataclass; dump known-safe subset
    d = {"data_path": cfg.data_path, "image_size": 256, "global_batch_size": 4, "world_size": 1}
    p.write_text(json.dumps(d), encoding="utf-8")
    loaded = TrainConfig(**json.loads(p.read_text(encoding="utf-8")))
    issues = validate_train_config(loaded, require_cuda=False)
    assert isinstance(issues, list)
