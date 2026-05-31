import numpy as np
from utils.modeling.hf_reward import HFRewardPanel, HFRewardWeights


def test_hf_reward_panel_empty_backends():
    panel = HFRewardPanel(
        weights=HFRewardWeights(hpsv2=0.0, pickscore=0.0, clip_h14=0.0, onealign=0.0, cafe_aesthetic=0.0)
    )
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    bd = panel.score_breakdown(rgb, prompt="test")
    assert bd.composite == 0.5
    assert bd.parts == {}


def test_hf_reward_panel_score_without_models():
    panel = HFRewardPanel(device="cpu")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    s = panel.score(rgb, prompt="a cat")
    assert 0.0 <= s <= 1.0
