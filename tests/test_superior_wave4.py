"""Wave 4 superior stack tests."""

from __future__ import annotations

import json
from pathlib import Path

from config.defaults.superior_stack import FlywheelPlan, SuperiorStackDefaults
from utils.superior.eval_report import build_markdown_report
from utils.superior.flywheel import run_flywheel
from utils.superior.hard_negative import merge_negative_prompt, mine_hard_negatives, tags_from_benchmark_row


def test_tags_from_benchmark_row_blur() -> None:
    tags = tags_from_benchmark_row({"composite": 0.4, "edge_sharpness": 50.0})
    assert "low_composite" in tags
    assert "blur" in tags


def test_mine_hard_negatives() -> None:
    rows = [
        {"composite": 0.3, "ocr_match": 0.2, "expected_text": "HELLO"},
        {"composite": 0.25, "edge_sharpness": 30.0},
    ]
    bundle = mine_hard_negatives(rows)
    assert bundle.negative_suffix
    merged = merge_negative_prompt("bad quality", bundle)
    assert "blurry" in merged.lower() or "text" in merged.lower()


def test_eval_report_markdown(tmp_path: Path) -> None:
    (tmp_path / "leaderboard.json").write_text(
        json.dumps([{"model": "base", "mean_composite": 0.8, "std_composite": 0.1, "robust_score": 0.75, "cases": 5}]),
        encoding="utf-8",
    )
    (tmp_path / "results.json").write_text(
        json.dumps([{"case": "c1", "composite": 0.4, "edge_sharpness": 40.0}]),
        encoding="utf-8",
    )
    md = build_markdown_report(tmp_path)
    assert "Leaderboard" in md
    assert "base" in md


def test_flywheel_dry_run() -> None:
    plan = FlywheelPlan(
        base_ckpt="fake.pt",
        work_dir="tmp_flywheel_test",
        skip_curate=True,
        defaults=SuperiorStackDefaults(auto_loop_iterations=1),
    )
    summary = run_flywheel(plan, dry_run=True)
    assert summary.get("status") == "ok"


def test_superior_preset_exists() -> None:
    from config.defaults.model_presets import PRESETS

    assert "superior" in PRESETS
    assert PRESETS["superior"].cfg_scale == 7.0
