import torch

from diffusion.cfg_schedulers import (
    cfg_scale_cosine_ramp,
    cfg_scale_linear,
    cfg_scale_piecewise,
    cfg_scale_snr_aware,
)
from diffusion.consistency_utils import (
    consistency_delta_loss,
    one_step_consistency_refine,
    temporal_ema_target,
)
from diffusion.self_conditioning import blend_self_cond, maybe_detached_self_cond


def test_cfg_schedules_are_bounded_and_finite():
    base = 7.5
    v0 = cfg_scale_linear(base, 0, 20, start_multiplier=0.7, end_multiplier=1.0)
    v1 = cfg_scale_cosine_ramp(base, 19, 20, min_multiplier=0.6, max_multiplier=1.0)
    v2 = cfg_scale_piecewise(base, 10, 30, stage_multipliers=(0.8, 1.0, 0.9))
    assert 0.0 < v0 <= base
    assert 0.0 < v1 <= base
    assert 0.0 < v2 <= base


def test_cfg_snr_aware_shape():
    a = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    out = cfg_scale_snr_aware(6.0, a)
    assert out.shape == a.shape
    assert torch.isfinite(out).all()


def test_self_condition_helpers():
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    sc = maybe_detached_self_cond(x, enabled=True, drop_prob=0.0, training=True)
    assert sc is not None and not sc.requires_grad
    y = blend_self_cond(x, sc, strength=0.3)
    assert y.shape == x.shape


def test_consistency_helpers():
    a = torch.randn(2, 4, 8, 8)
    b = torch.randn(2, 4, 8, 8)
    loss = consistency_delta_loss(a, b, reduction="mean")
    assert loss.ndim == 0 and torch.isfinite(loss)
    ema = temporal_ema_target(None, a, decay=0.9)
    assert ema.shape == a.shape
    refined = one_step_consistency_refine(a, b, step_size=0.2)
    assert refined.shape == a.shape
