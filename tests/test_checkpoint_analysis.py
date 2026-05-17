"""Tests for ``utils.checkpoint.checkpoint_manager.analyze_checkpoint_differences``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(slots=True)
class _SlottedTrainStub:
    lr: float


def test_analyze_checkpoint_differences_slotted_config(tmp_path: Path) -> None:
    """Embedded checkpoint configs may be slotted dataclasses (no ``__dict__``)."""
    torch = pytest.importorskip("torch")
    from utils.checkpoint.checkpoint_manager import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save(
        {"model": {"w": torch.tensor([1.0, 1.0])}, "config": _SlottedTrainStub(lr=0.01), "step": 0},
        p1,
    )
    torch.save(
        {"model": {"w": torch.tensor([2.0, 2.0])}, "config": _SlottedTrainStub(lr=0.02), "step": 1},
        p2,
    )
    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["statistics"]["step_difference"] == 1
    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["config_keys_differing"] == 1
    assert "lr" in out["config_differences"]
    assert out["config_differences"]["lr"]["checkpoint1"] == 0.01
    assert out["config_differences"]["lr"]["checkpoint2"] == 0.02


def test_analyze_checkpoint_single_el_tensor_no_std_warning(tmp_path: Path) -> None:
    """Single-element weight diffs must not trigger PyTorch unbiased-std warnings."""
    torch = pytest.importorskip("torch")
    from utils.checkpoint.checkpoint_manager import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.tensor([1.0])}, "config": {}, "step": 0}, p1)
    torch.save({"model": {"w": torch.tensor([2.0])}, "config": {}, "step": 0}, p2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyze_checkpoint_differences(str(p1), str(p2))

    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0
    assert out["parameter_differences"]["w"]["std_diff"] == 0.0
    dof_msgs = [str(w.message).lower() for w in caught if "degrees of freedom" in str(w.message).lower()]
    assert not dof_msgs, dof_msgs


def test_analyze_checkpoint_missing_model_dict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from utils.checkpoint.checkpoint_manager import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"step": 0, "config": {}}, p1)
    torch.save({"model": {"w": torch.tensor([1.0, 2.0])}, "step": 0, "config": {}}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["statistics"]["model_dict_checkpoint1"] == "missing_or_empty"
    assert out["statistics"]["model_dict_checkpoint2"] == "ok"
    assert out["parameter_differences"]["w"]["status"] == "only_in_checkpoint2"
    assert out["statistics"]["total_parameters"] == 0
    assert out["statistics"]["tensors_only_checkpoint2"] == 1
    assert out["statistics"]["tensors_compared"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0


def test_analyze_checkpoint_shape_mismatch_flagged(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from utils.checkpoint.checkpoint_manager import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "step": 0}, p1)
    torch.save({"model": {"w": torch.zeros(3)}, "step": 0}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["parameter_differences"]["w"]["status"] == "shape_mismatch"
    assert out["statistics"]["total_parameters"] == 0
    assert out["statistics"]["tensors_shape_mismatch"] == 1
    assert out["statistics"]["tensors_compared"] == 0
    assert out["statistics"]["tensor_keys_checked"] == 1
    assert out["statistics"]["tensors_skipped_non_tensor"] == 0


def test_analyze_checkpoint_non_tensor_weight_skipped(tmp_path: Path) -> None:
    """Model dicts may contain non-tensor values; diff must not call ``torch`` ops on them."""
    torch = pytest.importorskip("torch")
    from utils.checkpoint.checkpoint_manager import analyze_checkpoint_differences

    p1 = tmp_path / "a.pt"
    p2 = tmp_path / "b.pt"
    torch.save({"model": {"w": torch.zeros(2), "meta": 1}, "step": 0}, p1)
    torch.save({"model": {"w": torch.ones(2), "meta": 2}, "step": 0}, p2)

    out = analyze_checkpoint_differences(str(p1), str(p2))
    assert out["parameter_differences"]["meta"]["status"] == "skipped_non_tensor"
    assert out["parameter_differences"]["meta"]["type_checkpoint1"] == "int"
    assert out["statistics"]["tensors_skipped_non_tensor"] == 1
    assert out["statistics"]["tensors_compared"] == 1
    assert out["statistics"]["tensor_keys_checked"] == 2
    assert "w" in out["parameter_differences"]
    assert "mean_diff" in out["parameter_differences"]["w"]
