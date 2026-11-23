"""Detail Daemon (``--detail-amount``): model sees a scaled timestep mid-sampling, the solver keeps the real one."""

from __future__ import annotations

import math

import pytest
import torch
from diffusion.gaussian_diffusion import create_diffusion
from diffusion.solvers import detail_multipliers

SHAPE = (1, 4, 4, 4)


class TimeRecorder(torch.nn.Module):
    """Records the timestep of every call; output depends on ``t`` only when ``use_t``."""

    def __init__(self, use_t: bool = False):
        super().__init__()
        self.use_t, self.calls = use_t, []

    def forward(self, x, t, **_):
        self.calls.append(float(t[0]))
        scale = 0.1 + (t.float().view(-1, 1, 1, 1) / 1000.0 if self.use_t else 0.0)
        return x * scale


def _run(model, flow: bool, **kw):
    torch.manual_seed(0)
    extra = dict(flow_matching_sample=True, flow_solver="euler") if flow else dict(solver="ddim", scheduler="ddim")
    return create_diffusion().sample_loop(
        model, SHAPE, cfg_scale=1.0, num_inference_steps=10, device="cpu", **extra, **kw
    )


def test_schedule_shape():
    assert detail_multipliers(10, 0.0) == []
    m = detail_multipliers(21, 0.3, 0.2, 0.8)
    assert len(m) == 21
    assert m[:5] == [1.0] * 5 and m[-5:] == [1.0] * 5
    assert min(m) == pytest.approx(0.7) and m.index(min(m)) == 10
    assert m == pytest.approx(m[::-1])
    assert all(a >= b for a, b in zip(m[4:11], m[5:11], strict=False))
    assert max(detail_multipliers(21, -0.2)) == pytest.approx(1.2)
    assert min(detail_multipliers(21, 5.0)) == 0.0
    assert detail_multipliers(10, 0.3, 0.8, 0.2) == []


@pytest.mark.parametrize("flow", [False, True])
def test_zero_amount_is_bit_identical(flow):
    base = _run(TimeRecorder(use_t=True), flow)
    off = _run(TimeRecorder(use_t=True), flow, detail_amount=0.0, detail_start=0.0, detail_end=1.0)
    assert torch.equal(base, off)
    assert not torch.equal(base, _run(TimeRecorder(use_t=True), flow, detail_amount=0.3))


@pytest.mark.parametrize("flow", [False, True])
def test_model_gets_scaled_t_solver_keeps_real_t(flow):
    plain, daemon = TimeRecorder(), TimeRecorder()
    a = _run(plain, flow)
    b = _run(daemon, flow, detail_amount=0.3)
    mult = detail_multipliers(len(plain.calls), 0.3)
    assert daemon.calls == pytest.approx([t * m for t, m in zip(plain.calls, mult, strict=True)])
    assert any(not math.isclose(x, y) for x, y in zip(plain.calls, daemon.calls, strict=True))
    # A t-blind model gives the same trajectory: the solver's own coefficients still use the real timesteps.
    assert torch.equal(a, b)
