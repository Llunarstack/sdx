"""Superior stack wave tests (torch-dependent waves 5-12, merged)."""

from __future__ import annotations

# ---- Wave 5: research-backed DPO, FDG, feature cache, online reward. ----
import numpy as np
import torch
from utils.generation.cfg_batched import combine_cfg_outputs
from utils.superior.feature_cache import FeatureCacheConfig, FeatureCachePolicy
from utils.superior.frequency_cfg import apply_fdg_cfg, frequency_decoupled_cfg_delta
from utils.superior.inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv
from utils.superior.online_reward import OnlineRewardConfig, OnlineRewardModel
from utils.training.dpo_advanced import (
    safeguard_dpo_margins,
    safeguarded_dpo_preference_loss,
    timestep_dpo_weight,
)


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


def test_fdg_differs_from_standard_cfg() -> None:
    torch.manual_seed(0)
    cond = torch.randn(1, 4, 16, 16)
    uncond = torch.randn(1, 4, 16, 16)
    std = combine_cfg_outputs(cond, uncond, uncond, cfg_scale=7.5)
    fdg = apply_fdg_cfg(cond, uncond, cfg_scale=7.5, fdg_strength=1.0)
    assert not torch.allclose(std, fdg)


def test_frequency_decoupled_delta() -> None:
    delta = torch.randn(1, 4, 8, 8)
    out = frequency_decoupled_cfg_delta(delta, cfg_scale=7.0)
    assert out.shape == delta.shape


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
    rm = OnlineRewardModel(OnlineRewardConfig(vit_weight=0.0))
    s = rm.score_one(rgb, prompt="")
    assert 0.0 <= s <= 1.5


def test_inference_argv_fdg() -> None:
    cfg = SuperiorInferenceConfig(fdg_cfg_strength=0.65)
    argv = build_superior_sample_argv(ckpt="m.pt", prompt="cat", out="o.png", config=cfg)
    assert "--fdg-cfg-strength" in argv


# ---- Wave 6: block cache, flow-GRPO, consistency/LADD distill, hard-negative flywheel. ----

from utils.superior.block_cache import BlockCacheConfig, BlockDiTCache
from utils.superior.hard_negative import benchmark_sample_args_for_negatives, mine_hard_negatives
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


# ---- Wave 7: TaylorSeer, DenseGRPO, Rectified-CFG++, LCM sampling hooks. ----

from utils.generation.rectified_cfgpp import rectified_cfgpp_combine
from utils.superior.taylor_cache import TaylorBlockCache, TaylorCacheConfig, taylor_forecast_tensor
from utils.training.dense_grpo import (
    dense_reward_gains,
    estimate_x0_from_xt_flow,
    reward_aware_sde_scale,
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


def test_rectified_cfgpp_clamps_delta() -> None:
    cond = torch.randn(1, 4, 8, 8) * 10
    uncond = torch.zeros(1, 4, 8, 8)
    std = uncond + 7.5 * (cond - uncond)
    rc = rectified_cfgpp_combine(cond, uncond, cfg_scale=7.5, tangent_norm=0.5)
    assert not torch.allclose(std, rc)


def test_reward_aware_sde_peak_mid() -> None:
    mid = float(reward_aware_sde_scale(0.5, base=0.35))
    end = float(reward_aware_sde_scale(0.05, base=0.35))
    assert mid > end


def test_dense_reward_gains() -> None:
    gains = dense_reward_gains([0.8], [[0.2, 0.5, 0.8]])
    assert gains[0][0] == 0.2
    assert abs(gains[0][2] - 0.3) < 1e-6


def test_step_advantages_from_gains() -> None:
    adv = step_advantages_from_gains([0.1, 0.5, 0.2, 0.3])
    assert adv.numel() == 4


def test_flow_x0_estimate() -> None:
    x_t = torch.tensor([[[[1.0]]]])
    v = torch.tensor([[[[0.5]]]])
    t = torch.tensor([0.5])
    x0 = estimate_x0_from_xt_flow(x_t, v, t)
    assert float(x0.item()) == 0.75


# ---- Wave 8: APG guidance, Flash-GRPO, BranchGRPO. ----

from utils.generation.apg_guidance import apg_cfg_combine, decompose_parallel_orthogonal
from utils.training.branch_grpo import (
    BranchGRPOConfig,
    enumerate_branch_paths,
    fuse_branch_rewards,
    prefix_reuse_factor,
)
from utils.training.flash_grpo import (
    iso_temporal_group_advantages,
    rectify_policy_gradient,
    sde_discretization_lambda,
)


def test_apg_removes_parallel_oversaturation() -> None:
    cond = torch.tensor([[[[2.0, 1.0]]]])
    uncond = torch.tensor([[[[0.5, 0.5]]]])
    std = uncond + 7.5 * (cond - uncond)
    apg = apg_cfg_combine(cond, uncond, cfg_scale=7.5, parallel_eta=0.0)
    assert not torch.allclose(std, apg)
    delta = cond - uncond
    par, orth = decompose_parallel_orthogonal(delta, cond)
    assert torch.allclose(par + orth, delta)


def test_combine_cfg_apg_path() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    out = combine_cfg_outputs(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0)
    assert out.shape == cond.shape


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


def test_branch_paths_and_fusion() -> None:
    paths = enumerate_branch_paths(20, branch_factor=2, split_fractions=(0.35, 0.65))
    assert len(paths) == 4
    fused = fuse_branch_rewards([0.3, 0.9, 0.4, 0.8], mode="max")
    assert fused == 0.9
    save = prefix_reuse_factor(2, 2, 20)
    assert 0.0 <= save <= 1.0


def test_branch_config_defaults() -> None:
    cfg = BranchGRPOConfig()
    assert cfg.branch_factor >= 2


# ---- Wave 9: ZeResFDG, CFG-Zero*, QSilk micrograin. ----

from utils.generation.cfg_zero_star import cfg_zero_optimized_scale, cfg_zero_star_combine, zero_init_step_count
from utils.generation.guidance_stack import combine_guided_prediction
from utils.generation.micrograin_stabilizer import qsilk_micrograin_stabilize
from utils.generation.zeresfdg import (
    SpectralGuidanceEMA,
    apply_zeresfdg_cfg,
    energy_rescale_guided,
    zero_project_delta,
)


def test_zero_project_delta() -> None:
    uncond = torch.tensor([[[[1.0, 0.0]]]])
    delta = torch.tensor([[[[1.0, 1.0]]]])
    r = zero_project_delta(delta, uncond)
    dot = float((r * uncond).sum())
    assert abs(dot) < 1e-5


def test_zeresfdg_differs_from_cfg() -> None:
    cond = torch.randn(1, 4, 16, 16)
    uncond = torch.randn(1, 4, 16, 16)
    std = uncond + 7.0 * (cond - uncond)
    zr = apply_zeresfdg_cfg(cond, uncond, cfg_scale=7.0, cfg_rescale=0.7, strength=1.0)
    assert not torch.allclose(std, zr)


def test_spectral_ema_high_scale() -> None:
    ema = SpectralGuidanceEMA(detail_threshold=0.01)
    lat = torch.randn(1, 4, 8, 8)
    for _ in range(20):
        hs = ema.update(lat)
    assert hs >= ema.conservative_high_scale


def test_cfg_zero_star_zero_init() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    out = cfg_zero_star_combine(cond, uncond, cfg_scale=7.0, sample_step=0, total_steps=50)
    assert float(out.abs().max()) < 1e-6


def test_cfg_zero_optimized_scale() -> None:
    cond = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    uncond = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
    st = cfg_zero_optimized_scale(cond, uncond)
    assert float(st.mean()) > 0.5


def test_combine_guided_zeresfdg_priority() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=5.0,
        zeresfdg_strength=1.0,
        fdg_strength=0.65,
    )
    assert out.shape == cond.shape


def test_qsilk_stabilize_bounded() -> None:
    x = torch.randn(1, 4, 16, 16) * 5.0
    y = qsilk_micrograin_stabilize(x, detail_amount=0.1)
    assert y.abs().max() <= x.abs().max() + 0.5


def test_zero_init_step_count() -> None:
    assert zero_init_step_count(50, zero_init_frac=0.04) >= 1


def test_energy_rescale() -> None:
    guided = torch.randn(1, 4, 4, 4) * 3.0
    cond = torch.randn(1, 4, 4, 4)
    out = energy_rescale_guided(guided, cond)
    o = out.norm()
    c = cond.norm()
    assert abs(float(o - c) / float(c + 1e-6)) < 0.2


def test_combine_cfg_outputs_cfg_zero() -> None:
    cond = torch.randn(1, 2, 4, 4)
    uncond = torch.randn(1, 2, 4, 4)
    out = combine_cfg_outputs(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_zero_star=True,
        sample_step=0,
        total_steps=100,
    )
    assert float(out.abs().max()) < 1e-5


# ---- Wave 10: TP-GRPO, DyDiT dynamic width, APG momentum, linear attention. ----

from utils.generation.guidance_session import DynamicDitSchedule, GuidanceSession
from utils.superior.dynamic_dit import apply_dynamic_width, spatial_token_importance, timestep_dynamic_width
from utils.superior.linear_attention import hybrid_attention_fraction, linear_attention
from utils.training.turning_point_grpo import (
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


def test_guidance_session_momentum() -> None:
    sess = GuidanceSession(apg_momentum_beta=0.3)
    d = torch.ones(1, 2, 2, 2)
    sess.note_apg_delta(d)
    assert sess.prev_apg_delta is not None
    assert sess.step_index == 1


def test_combine_apg_with_session() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.randn(1, 4, 4, 4)
    sess = GuidanceSession(apg_momentum_beta=0.25)
    o1 = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0, guidance_session=sess)
    o2 = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, apg_parallel_eta=0.0, guidance_session=sess)
    assert o1.shape == o2.shape


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


def test_dynamic_dit_schedule() -> None:
    sched = DynamicDitSchedule(enabled=True, early_width=0.85, late_width=1.0)
    assert sched.scale_at_progress(0.0) < sched.scale_at_progress(1.0)


# ---- Wave 11: Wave 12: CFG++, interval CFG, GRPO-Guard, CFG-Rejection, branch rollout. ----

from utils.generation.cfg_interval import should_apply_cfg
from utils.generation.cfg_pp import cfg_pp_combine, cfg_scale_to_pp_lambda
from utils.superior.cfg_rejection import CFGRejectionTracker, pick_best_candidate_index
from utils.training.branch_grpo import branch_rollout_flow_samples
from utils.training.grpo_guard import GRPOGuardConfig, grpo_guard_weighted_loss, ratio_norm_advantages


def test_cfg_pp_differs_from_standard_cfg() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    pp = cfg_pp_combine(cond, uncond, cfg_lambda=0.6)
    std = uncond + 7.0 * (cond - uncond)
    assert not torch.allclose(pp, std)


def test_cfg_scale_to_pp_lambda() -> None:
    assert 0.0 < cfg_scale_to_pp_lambda(7.5) < 1.0


def test_interval_cfg_skips_early() -> None:
    assert not should_apply_cfg(0.05, skip_early_frac=0.15)
    assert should_apply_cfg(0.5, skip_early_frac=0.15)


def test_combine_guided_cfg_pp() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.zeros_like(cond)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_pp_lambda=0.55,
        zeresfdg_strength=0.0,
    )
    assert out.shape == cond.shape


def test_interval_skips_in_stack() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_skip_early_frac=0.5,
        sample_step=0,
        total_steps=10,
    )
    assert torch.allclose(out, cond)


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


# ---- Wave 12: Wave 13: TCFG, SLG, DBCache fingerprint, guidance probe. ----

from utils.generation.guidance_probe import GuidanceProbe
from utils.generation.slg_guidance import parse_skip_blocks, slg_combine
from utils.generation.tcfg import tcfg_combine, tcfg_damp_unconditional


def test_tcfg_damp_changes_uncond() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = cond + torch.randn_like(cond) * 0.5
    damped = tcfg_damp_unconditional(cond, uncond, damping=1.0)
    assert not torch.allclose(damped, uncond)


def test_tcfg_combine_differs_from_cfg() -> None:
    cond = torch.randn(1, 4, 4, 4)
    uncond = torch.randn(1, 4, 4, 4)
    out = tcfg_combine(cond, uncond, cfg_scale=7.0, damping=0.9)
    plain = uncond + 7.0 * (cond - uncond)
    assert not torch.allclose(out, plain)


def test_combine_guided_tcfg() -> None:
    cond = torch.randn(1, 2, 2, 2)
    uncond = torch.zeros_like(cond)
    out = combine_guided_prediction(cond, uncond, uncond, cfg_scale=5.0, tcfg_damping=0.8, zeresfdg_strength=0.0)
    assert out.shape == cond.shape


def test_slg_combine() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    skip = cond * 0.5
    out = slg_combine(cond, uncond, skip, cfg_scale=7.0, slg_scale=2.0)
    assert out.shape == cond.shape


def test_parse_skip_blocks_auto() -> None:
    blocks = parse_skip_blocks("auto", depth=12)
    assert len(blocks) >= 1


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
