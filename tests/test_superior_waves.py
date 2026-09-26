"""Superior stack wave tests (torch-free waves 3-4, merged)."""

from __future__ import annotations

# ---- Wave 3: superior stack tests. ----
from pathlib import Path

from utils.superior.auto_loop import AutoImproveConfig, build_auto_improve_argv
from utils.superior.prompt_expand import expand_prompt_heuristic
from utils.superior.vit_mining import ViTMineConfig, blended_reward, mine_vit_preference_pairs


def test_blended_reward() -> None:
    assert blended_reward(0.8, 0.6, vit_weight=0.5) == 0.7


def test_mine_vit_pairs_without_vit_scores() -> None:
    rows = [
        {"case": "c", "prompt": "cat", "output": "a.png", "composite": 0.9},
        {"case": "c", "prompt": "cat", "output": "b.png", "composite": 0.4},
    ]

    def _fake_score(path, prompt, **kw):
        return (0.5, 0.9 if "a.png" in str(path) else 0.3)

    import utils.superior.vit_mining as vm

    orig = vm.score_image_vit
    vm.score_image_vit = _fake_score
    try:
        cfg = ViTMineConfig(vit_ckpt="fake.pt", min_margin=0.1)
        pairs = mine_vit_preference_pairs(rows, cfg)
        assert len(pairs) >= 1
        assert pairs[0]["source"] == "vit_mining"
    finally:
        vm.score_image_vit = orig


def test_expand_prompt_heuristic() -> None:
    out = expand_prompt_heuristic("a red fox in snow")
    assert "detail" in out.lower() or "lighting" in out.lower()


def test_build_auto_improve_argv() -> None:
    cfg = AutoImproveConfig(
        base_ckpt="base.pt",
        vit_ckpt="vit/best.pt",
        local_rag_jsonl="facts.jsonl",
        model_soup=True,
    )
    argv = build_auto_improve_argv(cfg)
    assert "--vit-ckpt" in argv
    assert "--local-rag-jsonl" in argv
    assert "--model-soup" in argv
    assert "superior_composite" in argv


def test_benchmark_vit_mine_path(tmp_path: Path) -> None:
    from scripts.tools.benchmark_suite import _write_preference_jsonl

    rows = [
        {"case": "c", "prompt": "x", "output": "w.png", "composite": 0.9},
        {"case": "c", "prompt": "x", "output": "l.png", "composite": 0.3},
    ]
    import utils.superior.vit_mining as vm

    orig = vm.mine_vit_preference_pairs
    vm.mine_vit_preference_pairs = lambda r, c: [
        {"win_image_path": "w.png", "lose_image_path": "l.png", "caption": "x"}
    ]
    try:
        n = _write_preference_jsonl(
            tmp_path / "p.jsonl",
            rows,
            min_margin=0.05,
            max_pairs_per_case=1,
            vit_ckpt="v.pt",
            vit_mine=True,
        )
        assert n == 1
    finally:
        vm.mine_vit_preference_pairs = orig


# ---- Wave 4: superior stack tests. ----

import json

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
