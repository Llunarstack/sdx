"""Wave 5: research-backed DPO, FDG, feature cache, online reward."""

from __future__ import annotations

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
