"""Device / perf helpers (no GPU required for string/dataclass checks)."""

from __future__ import annotations

from utils.training.device_perf import parallel_train_torchrun_example, training_perf_hints


def test_parallel_train_torchrun_example_shape() -> None:
    s = parallel_train_torchrun_example(3, extra="--data-path x")
    assert "--standalone" in s and "--nproc_per_node=3" in s
    assert "--data-path x" in s and s.strip().startswith("python")


def test_training_perf_hints_keys() -> None:
    h = training_perf_hints(num_workers=4, world_size=2)
    assert "world_size=2" in h.ddp_note
    assert "expandable_segments" in h.vram_fragmentation_note
    assert "num_workers=4" in h.datloader_note
