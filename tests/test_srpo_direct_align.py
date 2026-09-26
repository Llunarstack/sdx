import pytest

torch = pytest.importorskip("torch")

from utils.training.srpo_direct_align import (  # noqa: E402
    SRPOConfig,
    direct_align_recover_flow,
    direct_align_recover_vp,
    semantic_relative_reward,
    srpo_advantage_weights,
    timestep_reward_mask,
)


def test_flow_recovery_is_exact_on_true_velocity():
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x0)
    s = 0.7
    x_s = (1 - s) * x0 + s * eps
    v = eps - x0
    rec = direct_align_recover_flow(x_s, v, s)
    assert torch.allclose(rec, x0, atol=1e-5)


def test_vp_recovery_is_exact_on_true_eps():
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x0)
    ab = torch.tensor([0.5, 0.9])
    ab_e = ab.view(-1, 1, 1, 1)
    x_t = ab_e.sqrt() * x0 + (1 - ab_e).sqrt() * eps
    rec = direct_align_recover_vp(x_t, eps, ab)
    assert torch.allclose(rec, x0, atol=1e-5)


def test_semantic_relative_reward_cancels_global_bias():
    # A reward with a constant global offset: relative scoring removes it.
    def biased_score(images, text):
        base = torch.full((images.shape[0],), 10.0)
        return base + (1.0 if "realistic" in text else 0.0)

    cfg = SRPOConfig(control_positive="realistic", control_negative="plastic")
    r = semantic_relative_reward(biased_score, torch.zeros(3, 3, 8, 8), "portrait", config=cfg)
    assert torch.allclose(r, torch.ones(3))


def test_timestep_mask_zeroes_late_steps():
    fr = torch.tensor([0.1, 0.5, 0.8, 0.95])
    mask = timestep_reward_mask(fr, cutoff=0.75)
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_advantage_weights_prefer_high_reward_and_mask_late():
    rewards = torch.tensor([1.0, -1.0, 2.0, 0.0])
    fractions = torch.tensor([0.2, 0.2, 0.9, 0.2])
    w = srpo_advantage_weights(rewards, fractions)
    assert w[0] < w[1]  # higher reward -> lower loss weight (reinforced)
    assert w[2] == 0.0  # late-timestep contribution masked
    assert (w >= 0).all()
