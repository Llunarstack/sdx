"""LCM flow-vs-VP sampling branch, AMP/search CLI flags, and search helper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_main import lcm_should_enable_flow_sample, resolve_inference_amp
from utils.generation.simple_latent_generate import sample_with_test_time_search


def test_lcm_does_not_force_flow_on_vp_checkpoint() -> None:
    dit = SimpleNamespace(flow_matching_training=False)
    lcm = SimpleNamespace(flow_matching_training=False)
    assert lcm_should_enable_flow_sample(dit, lcm) is False
    assert lcm_should_enable_flow_sample(dit, None) is False


def test_lcm_enables_flow_when_dit_is_flow_trained() -> None:
    dit = SimpleNamespace(flow_matching_training=True)
    lcm = SimpleNamespace(flow_matching_training=False)
    assert lcm_should_enable_flow_sample(dit, lcm) is True


def test_lcm_enables_flow_when_overlay_is_flow_trained() -> None:
    dit = SimpleNamespace(flow_matching_training=False)
    lcm = SimpleNamespace(flow_matching_training=True)
    assert lcm_should_enable_flow_sample(dit, lcm) is True


def test_inference_amp_and_search_cli_flags() -> None:
    parser = build_sample_parser()
    args = parser.parse_args(["--ckpt", "x.pt", "--prompt", "a"])
    assert args.inference_amp == "auto"
    assert args.test_time_search is False
    assert args.test_time_search_pool == 8
    assert args.uncensored_mode is True
    args_safe = parser.parse_args(["--ckpt", "x.pt", "--no-uncensored-mode"])
    assert args_safe.uncensored_mode is False
    args2 = parser.parse_args(
        [
            "--ckpt",
            "x.pt",
            "--inference-amp",
            "bf16",
            "--test-time-search",
            "--test-time-search-pool",
            "4",
        ]
    )
    assert args2.inference_amp == "bf16"
    assert args2.test_time_search is True
    assert args2.test_time_search_pool == 4
    args3 = parser.parse_args(["--ckpt", "x.pt", "--inference-amp", "off"])
    assert args3.inference_amp == "off"
    amp_help = parser.format_help()
    assert "torch.cuda.is_bf16_supported()" in amp_help


def test_resolve_inference_amp_modes() -> None:
    cpu = SimpleNamespace(type="cpu")
    assert resolve_inference_amp(SimpleNamespace(inference_amp="off"), cpu) == (False, None)
    on, dtype = resolve_inference_amp(SimpleNamespace(inference_amp="fp16"), cpu)
    assert on is True and dtype is torch.float16
    on_bf, dtype_bf = resolve_inference_amp(SimpleNamespace(inference_amp="bf16"), cpu)
    assert on_bf is True and dtype_bf is torch.bfloat16
    auto_cpu, auto_dt = resolve_inference_amp(SimpleNamespace(inference_amp="auto"), cpu)
    assert auto_cpu is False and auto_dt is None


def test_sample_with_test_time_search_picks_best_seed() -> None:
    def sample_fn(seed: int, steps: int) -> np.ndarray:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[..., 0] = min(255, int(seed) * 20 + int(steps))
        return img

    def score_fn(images) -> list[float]:
        return [float(np.asarray(im)[..., 0].mean()) for im in images]

    winner = sample_with_test_time_search(
        sample_fn=sample_fn,
        score_fn=score_fn,
        pool_size=4,
        final_steps=12,
        base_seed=0,
        prompt="x",
        pick_metric="combo",
        device="cpu",
    )
    assert winner.shape == (8, 8, 3)
    assert winner.dtype == np.uint8
    # Later seeds have higher red channel; tournament should keep a high-seed image.
    assert int(winner[..., 0].mean()) >= int(sample_fn(0, 12)[..., 0].mean())
