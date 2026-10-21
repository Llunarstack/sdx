"""SRPO realism reward + late-timestep mask (``utils.training.flow_grpo``, wired via ``train_flow_grpo --srpo``)."""

import pytest

torch = pytest.importorskip("torch")

from utils.training.flow_grpo import SRPOConfig, semantic_relative_reward, timestep_reward_mask  # noqa: E402


def test_semantic_relative_reward_cancels_global_bias():
    def biased_score(images, text):
        return torch.full((images.shape[0],), 10.0) + (1.0 if "realistic" in text else 0.0)

    cfg = SRPOConfig(control_positive="realistic", control_negative="plastic")
    r = semantic_relative_reward(biased_score, torch.zeros(3, 3, 8, 8), "portrait", config=cfg)
    assert torch.allclose(r, torch.ones(3))


def test_semantic_relative_reward_works_on_float_scores():
    cfg = SRPOConfig(control_positive="natural", control_negative="waxy")
    assert semantic_relative_reward(lambda img, text: 0.7 if "natural" in text else 0.4, None, "", config=cfg) == (
        pytest.approx(0.3)
    )


def test_timestep_mask_zeroes_late_steps():
    mask = timestep_reward_mask(torch.tensor([0.1, 0.5, 0.8, 0.95]), cutoff=0.75)
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]
