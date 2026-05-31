"""Wave 3 superior stack tests."""

from __future__ import annotations

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
