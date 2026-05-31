"""Superior stack: DPO, soup, quality gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from utils.superior.dpo_pipeline import MinePairsConfig, build_dpo_train_argv, mine_pairs_from_benchmark
from utils.superior.model_soup import average_state_dicts
from utils.superior.quality_gates import QualityGateRunner


def test_mine_pairs_from_benchmark(tmp_path: Path) -> None:
    bench = tmp_path / "bench.json"
    w = tmp_path / "w.png"
    loser = tmp_path / "l.png"
    w.write_bytes(b"x")
    loser.write_bytes(b"y")
    bench.write_text(
        json.dumps(
            [
                {"case": "c1", "prompt": "a cat", "output": str(w), "composite": 0.9},
                {"case": "c1", "prompt": "a cat", "output": str(loser), "composite": 0.5},
            ]
        ),
        encoding="utf-8",
    )
    cfg = MinePairsConfig(benchmark_json=str(bench), out_jsonl=str(tmp_path / "p.jsonl"), min_margin=0.1)
    pairs = mine_pairs_from_benchmark(cfg)
    assert len(pairs) == 1
    assert pairs[0]["win_image_path"].endswith("w.png")


def test_build_dpo_argv() -> None:
    from utils.superior.dpo_pipeline import DPOStageConfig

    argv = build_dpo_train_argv(DPOStageConfig(ckpt="m.pt", preference_jsonl="p.jsonl"))
    assert "--ckpt" in argv and "m.pt" in argv


def test_average_state_dicts() -> None:
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([3.0, 4.0])}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 3.0]))


def test_quality_gate_runner() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    img[10:20, 10:20, 0] = 255
    runner = QualityGateRunner()
    res = runner.evaluate(img, prompt="test")
    assert "sharpness" in res.scores
