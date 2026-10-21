"""Superior stack: DPO, soup, quality gates.

Also holds what used to be: test_superior_extended, test_superior_stack, test_superior_waves, test_superior_waves_torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from utils.superior.auto_loop import (
    AutoImproveConfig,
    MinePairsConfig,
    ViTMineConfig,
    average_state_dicts,
    blended_reward,
    build_auto_improve_argv,
    build_dpo_train_argv,
    mine_pairs_from_benchmark,
    mine_vit_preference_pairs,
)
from utils.superior.caches import FeatureCacheConfig, FeatureCachePolicy
from utils.superior.inference_pipeline import (
    CompositeRanker,
    QualityGateRunner,
    SelfCorrectConfig,
    SelfCorrectPolicy,
    SuperiorInferenceConfig,
    SuperiorPromptStack,
    TfidfFactIndex,
    build_superior_sample_argv,
    build_tfidf_index_from_jsonl,
    expand_prompt_heuristic,
)
from utils.training.preference import (
    safeguard_dpo_margins,
    safeguarded_dpo_preference_loss,
    timestep_dpo_weight,
)

# ==================================================================================================
# test_superior_extended: Superior stack: DPO, soup, quality gates.
# ==================================================================================================


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
    from utils.superior.auto_loop import DPOStageConfig

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


# ==================================================================================================
# test_superior_stack: Tests for utils/superior stack.
# ==================================================================================================


def test_tfidf_retrieval_orders_relevant_facts() -> None:
    idx = TfidfFactIndex(
        facts=[
            "neon signs reflect on wet pavement in cyberpunk alleys",
            "pastoral meadow with sheep at sunrise",
            "database indexing with B-trees",
        ]
    )
    hits = idx.query("cyberpunk alley neon rain", top_k=2)
    assert "cyberpunk" in hits[0].lower()


def test_superior_prompt_stack_no_rag() -> None:
    out = SuperiorPromptStack(append_domain_tips=False).enrich("a red fox")
    assert out == "a red fox"


def test_build_superior_sample_argv() -> None:
    cfg = SuperiorInferenceConfig(num_candidates=4, local_rag_jsonl="facts.jsonl")
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert "--pick-best" in argv
    assert "superior_composite" in argv
    assert "--local-rag-jsonl" in argv


def test_composite_ranker_picks_sharper_image() -> None:
    blur = np.full((32, 32, 3), 128, dtype=np.uint8)
    sharp = blur.copy()
    sharp[8:24, 8:24] = 255
    sharp[8:24, 8:24:1] = 0
    ranker = CompositeRanker({"edge": 1.0, "exposure": 0.5})
    idx, scores = ranker.pick_best_index([blur, sharp], prompt="test", device="cpu")
    assert idx == 1 and scores[1] > scores[0]


def test_self_correct_policy_threshold() -> None:
    pol = SelfCorrectPolicy(SelfCorrectConfig(align_threshold=0.4))
    assert pol.needs_correction(0.2)
    assert not pol.needs_correction(0.9)


def test_build_index_missing_file() -> None:
    idx = build_tfidf_index_from_jsonl("/nonexistent/path.jsonl")
    assert idx.n_docs == 0


# ==================================================================================================
# test_superior_waves: Superior stack wave tests (torch-free waves 3-4, merged).
# ==================================================================================================

# ---- Wave 3: superior stack tests. ----


def test_blended_reward() -> None:
    assert blended_reward(0.8, 0.6, vit_weight=0.5) == 0.7


def test_mine_vit_pairs_without_vit_scores() -> None:
    rows = [
        {"case": "c", "prompt": "cat", "output": "a.png", "composite": 0.9},
        {"case": "c", "prompt": "cat", "output": "b.png", "composite": 0.4},
    ]

    def _fake_score(path, prompt, **kw):
        return (0.5, 0.9 if "a.png" in str(path) else 0.3)

    import utils.superior.auto_loop as vm

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
    import utils.superior.auto_loop as vm

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


from config.defaults.superior_stack import FlywheelPlan, SuperiorStackDefaults
from utils.superior.auto_loop import build_markdown_report, mine_hard_negatives, run_flywheel, tags_from_benchmark_row


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
    suffix = bundle.negative_suffix.lower()
    assert "blurry" in suffix or "text" in suffix


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


# ==================================================================================================
# test_superior_waves_torch: Superior stack wave tests (torch-dependent waves 5-12, merged).
# ==================================================================================================

# ---- Wave 5: research-backed DPO, feature cache, online reward. ----


def test_timestep_dpo_weight_high_noise() -> None:
    t = torch.tensor([0, 500, 999], dtype=torch.long)
    w = timestep_dpo_weight(t, 1000, mode="high_noise", power=0.5)
    assert w[2] > w[0]


def test_safeguard_shrinks_loser_margin() -> None:
    win = torch.tensor([2.0, 1.0])
    lose = torch.tensor([0.5, 0.4])
    lw, ll = safeguard_dpo_margins(win, lose, strength=0.85)
    assert (ll > lose).all()
    assert (lw == win).all()


def test_safeguarded_dpo_with_weights() -> None:
    win = torch.tensor([1.0, 0.8])
    lose = torch.tensor([0.2, 0.1])
    rw = torch.tensor([1.0, 0.9])
    rl = torch.tensor([0.8, 0.7])
    tw = torch.tensor([1.0, 2.0])
    loss = safeguarded_dpo_preference_loss(win, lose, rw, rl, beta=10.0, safeguard_strength=0.85, timestep_weights=tw)
    assert torch.isfinite(loss)


def test_feature_cache_reuse() -> None:
    pol = FeatureCachePolicy(FeatureCacheConfig(delta_threshold=0.5, max_reuse_streak=2))
    x = torch.zeros(1, 4, 4, 4)
    pred = torch.ones(1, 4, 4, 4)
    pol.note_fresh(x, pred)
    x2 = x + 0.01
    assert pol.should_skip_forward(x2)
    cached = pol.cached_prediction()
    assert torch.allclose(cached, pred)


def test_online_reward_scores() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[20:40, 20:40] = 200
    s = CompositeRanker({"ai_look": 0.5, "art": 0.5}).score_one(rgb, prompt="", device="cpu")
    assert 0.0 <= s <= 1.0


def test_inference_argv_fdg() -> None:
    cfg = SuperiorInferenceConfig(fdg_low=0.5, fdg_high=1.5)
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert argv[argv.index("--fdg-high") + 1] == "1.5"


# ---- Wave 6: block cache, flow-GRPO, consistency/LADD distill, hard-negative flywheel. ----

from utils.superior.auto_loop import benchmark_sample_args_for_negatives
from utils.superior.caches import BlockCacheConfig, BlockDiTCache
from utils.training.flow_grpo import group_relative_advantages, grpo_weighted_loss, reference_kl_penalty


def test_block_cache_skip_and_apply() -> None:
    cache = BlockDiTCache(BlockCacheConfig(rel_l1_threshold=0.5, recompute_every=99))
    fp1 = torch.tensor([1.0, 2.0, 3.0])
    fp2 = torch.tensor([1.01, 2.01, 3.01])
    cache.begin_forward(fp1)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp2)
    assert cache.should_skip_block(0)
    out = cache.apply_residual(x, 0)
    assert torch.allclose(out, x + 1.0)


def test_grpo_advantages_zero_mean() -> None:
    r = torch.tensor([0.2, 0.5, 0.9, 0.3])
    adv = group_relative_advantages(r)
    assert abs(float(adv.mean())) < 0.01


def test_grpo_weighted_loss() -> None:
    loss = torch.tensor([1.0, 2.0, 3.0])
    adv = torch.tensor([1.0, 0.0, -1.0])
    w = grpo_weighted_loss(loss, adv)
    assert torch.isfinite(w)


def test_reference_kl_penalty() -> None:
    a = torch.randn(1, 4, 4, 4)
    b = a + 0.1
    p = reference_kl_penalty(a, b, coef=0.1)
    assert float(p) > 0.0


def test_hard_negative_benchmark_args() -> None:
    bundle = mine_hard_negatives([{"composite": 0.3, "edge_sharpness": 20.0}])
    args = benchmark_sample_args_for_negatives(bundle)
    assert "--negative-prompt" in args
    assert args[1]


# ---- Wave 7: TaylorSeer, DenseGRPO, LCM sampling hooks. ----

from utils.superior.caches import TaylorBlockCache, TaylorCacheConfig, taylor_forecast_tensor
from utils.training.grpo import (
    estimate_x0_from_xt_flow,
    step_advantages_from_gains,
)


def test_taylor_forecast_linear() -> None:
    h0 = torch.ones(2, 2)
    h1 = h0 + 0.5
    pred = taylor_forecast_tensor([h0, h1], steps_since_anchor=2, interval=4, max_order=1)
    assert pred.shape == h0.shape
    assert float(pred.mean()) > float(h1.mean())


def test_taylor_block_cache_apply() -> None:
    cache = TaylorBlockCache(TaylorCacheConfig(rel_l1_threshold=0.5, recompute_every=8))
    fp = torch.tensor([1.0, 2.0])
    cache.begin_forward(fp)
    x = torch.zeros(1, 4, 2, 2)
    cache.note_block(0, x, x + 1.0)
    cache.begin_forward(fp + 0.001)
    cache.begin_forward(fp + 0.002)
    out = cache.apply_residual(x, 0)
    assert out.shape == x.shape


def test_step_advantages_from_gains() -> None:
    adv = step_advantages_from_gains([0.1, 0.5, 0.2, 0.3])
    assert adv.numel() == 4


def test_flow_x0_estimate() -> None:
    x_t = torch.tensor([[[[1.0]]]])
    v = torch.tensor([[[[0.5]]]])
    t = torch.tensor([0.5])
    x0 = estimate_x0_from_xt_flow(x_t, v, t)
    assert float(x0.item()) == 0.75


# ---- Wave 8: Flash-GRPO, BranchGRPO. ----

from utils.training.grpo import (
    BranchGRPOConfig,
    enumerate_branch_paths,
    iso_temporal_group_advantages,
    rectify_policy_gradient,
    sde_discretization_lambda,
)


def test_sde_lambda_peaks_mid() -> None:
    mid = float(sde_discretization_lambda(0.5))
    edge = float(sde_discretization_lambda(0.02))
    assert mid > edge


def test_iso_temporal_advantages() -> None:
    rewards = torch.tensor([0.9, 0.8, 0.2, 0.1])
    t_idx = torch.tensor([10, 10, 20, 20])
    adv = iso_temporal_group_advantages(rewards, t_idx, num_timesteps=50)
    assert adv[0] > adv[1]
    assert adv[2] > adv[3]


def test_rectify_policy_gradient() -> None:
    loss = torch.tensor(1.0)
    rect = rectify_policy_gradient(loss, 0.5)
    assert float(rect) > float(loss)


def test_branch_paths() -> None:
    paths = enumerate_branch_paths(20, branch_factor=2, split_fractions=(0.35, 0.65))
    assert len(paths) == 4


def test_branch_config_defaults() -> None:
    cfg = BranchGRPOConfig()
    assert cfg.branch_factor >= 2


# ---- Wave 9: QSilk micrograin. ----

from utils.generation.latent_filters import qsilk_micrograin_stabilize


def test_qsilk_stabilize_bounded() -> None:
    x = torch.randn(1, 4, 16, 16) * 5.0
    y = qsilk_micrograin_stabilize(x, detail_amount=0.1)
    assert y.abs().max() <= x.abs().max() + 0.5


# ---- Wave 10: TP-GRPO, DyDiT dynamic width, linear attention. ----

from models.linear_attention import hybrid_attention_fraction, linear_attention
from utils.superior.caches import apply_dynamic_width, spatial_token_importance, timestep_dynamic_width
from utils.training.grpo import (
    TurningPointGRPOConfig,
    detect_turning_point_indices,
    incremental_rewards_from_trajectory,
    tp_grpo_step_weights,
)


def test_incremental_rewards() -> None:
    inc = incremental_rewards_from_trajectory([0.2, 0.5, 0.9])
    assert inc == [0.2, 0.3, 0.4]


def test_turning_point_detection() -> None:
    inc = [0.1, -0.2, -0.1, 0.3]
    tps = detect_turning_point_indices(inc)
    assert 0 in tps


def test_tp_grpo_weights_boost_turning_point() -> None:
    traj = [0.1, 0.3, 0.55, 0.5, 0.9]
    w = tp_grpo_step_weights(traj, 0.9, config=TurningPointGRPOConfig(long_term_weight=0.5))
    assert len(w) == len(traj)
    assert max(w) > min(w)


def test_timestep_dynamic_width_monotone() -> None:
    early = timestep_dynamic_width(0.0, early=0.8, late=1.0)
    late = timestep_dynamic_width(1.0, early=0.8, late=1.0)
    assert early < late


def test_apply_dynamic_width() -> None:
    out = torch.ones(1, 4, 2, 2)
    scaled = apply_dynamic_width(out, 0.0, early=0.5)
    assert float(scaled.mean()) == 0.5


def test_spatial_importance_shape() -> None:
    x = torch.randn(2, 4, 8, 8)
    imp = spatial_token_importance(x)
    assert imp.shape == (2, 1, 8, 8)


def test_linear_attention_runs() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = linear_attention(q, k, v)
    assert out.shape == v.shape


def test_hybrid_attention_blend() -> None:
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    out = hybrid_attention_fraction(q, k, v, linear_frac=0.5)
    assert out.shape == v.shape


# ---- Wave 11: GRPO-Guard, CFG-Rejection, branch rollout. ----

from utils.superior.caches import CFGRejectionTracker, pick_best_candidate_index
from utils.training.grpo import (
    GRPOGuardConfig,
    branch_rollout_flow_samples,
    grpo_guard_weighted_loss,
    ratio_norm_advantages,
)


def test_grpo_guard_weights() -> None:
    adv = torch.tensor([1.0, -1.0, 0.2])
    normed = ratio_norm_advantages(adv)
    assert normed.numel() == 3


def test_grpo_guard_loss() -> None:
    loss = torch.tensor([0.5, 0.8])
    adv = torch.tensor([1.0, -0.5])
    t = torch.tensor([0.3, 0.7])
    total = grpo_guard_weighted_loss(loss, adv, t, config=GRPOGuardConfig())
    assert float(total.item()) > 0.0


def test_cfg_rejection_tracker() -> None:
    tr = CFGRejectionTracker(tau_steps=2)
    c = torch.randn(1, 4, 4, 4)
    u = torch.zeros_like(c)
    tr.note(c, u)
    tr.note(c * 2, u)
    assert tr.accumulated_early_score() > 0.0


def test_pick_best_candidate() -> None:
    assert pick_best_candidate_index([0.9, 0.2, 0.5]) == 1


def test_branch_rollout_paths() -> None:
    calls: list[int] = []

    def _roll() -> torch.Tensor:
        calls.append(1)
        return torch.zeros(1, 2, 2, 2)

    outs = branch_rollout_flow_samples(_roll, num_paths=4, steps=8, base_seed=0)
    assert len(outs) == 4
    assert len(calls) == 4
    assert len(enumerate_branch_paths(8)) >= 1


# ---- Wave 12: DBCache fingerprint, guidance probe. ----

from utils.generation.guidance_probe import GuidanceProbe


def test_guidance_probe_rerank() -> None:
    probe = GuidanceProbe(tau_steps=2)
    c = torch.randn(2, 4, 4, 4)
    u = torch.zeros_like(c)
    probe.note(c, u, step=0)
    probe.note(c * 2, u, step=1)
    order = probe.rerank_indices()
    assert len(order) == 2


def test_block_cache_cfg_split_fingerprint() -> None:
    cache = BlockDiTCache()
    t = torch.randn(4, 256)
    x = torch.randn(4, 16, 64)
    fp_full = cache.fingerprint_from_tensors(t, x, cfg_split=False)
    fp_split = cache.fingerprint_from_tensors(t, x, cfg_split=True)
    assert fp_split.shape == fp_full.shape
