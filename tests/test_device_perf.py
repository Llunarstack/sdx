"""Device / perf helpers (no GPU required for string/dataclass checks)."""

from __future__ import annotations

from utils.training.device_perf import parallel_train_torchrun_example, training_perf_hints
from utils.training.fast_dataloader import dataloader_perf_kwargs, resolve_training_num_workers
from utils.training.throughput import encode_text_multi_group


def test_parallel_train_torchrun_example_shape() -> None:
    s = parallel_train_torchrun_example(3, extra="--data-path x")
    assert "--standalone" in s and "--nproc_per_node=3" in s
    assert "--data-path x" in s and s.strip().startswith("python")


def test_training_perf_hints_keys() -> None:
    h = training_perf_hints(num_workers=4, world_size=2)
    assert "world_size=2" in h.ddp_note
    assert "expandable_segments" in h.vram_fragmentation_note
    assert "num_workers=4" in h.datloader_note


def test_dataloader_perf_kwargs_prefetch() -> None:
    kw = dataloader_perf_kwargs(num_workers=4, prefetch_factor=3)
    assert kw["num_workers"] == 4
    assert kw["prefetch_factor"] == 3
    assert kw["pin_memory"] is False or kw["pin_memory"] is True
    kw0 = dataloader_perf_kwargs(num_workers=0, prefetch_factor=2)
    assert "prefetch_factor" not in kw0


def test_resolve_training_num_workers_auto() -> None:
    assert resolve_training_num_workers(-1, 10_000, 32, auto=False) >= 2
    assert resolve_training_num_workers(6, 10_000, 32, auto=False) == 6


def test_encode_text_multi_group() -> None:
    calls: list = []

    def _fake_encode(caps: list) -> list:
        calls.append(list(caps))
        return [[len(c)] for c in caps]

    g0, g1, g2 = encode_text_multi_group([["p1", "p2"], None, ["s1"]], _fake_encode)
    assert calls == [["p1", "p2", "s1"]]
    assert g0 == [[2], [2]]
    assert g1 is None
    assert g2 == [[2]]
