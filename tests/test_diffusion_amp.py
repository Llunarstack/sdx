"""Inference AMP kwargs and sampling-loop CPU smoke tests."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
from diffusion.gaussian_diffusion import GaussianDiffusion, create_diffusion
from utils.generation.cfg_batched import batched_cfg_forward


class _ConstVel(nn.Module):
    def forward(self, x, t, **kwargs):
        del t, kwargs
        return torch.zeros_like(x)


class _AutocastProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.saw_autocast = False

    def forward(self, x, t, **kwargs):
        del t, kwargs
        self.saw_autocast = bool(torch.is_autocast_enabled())
        return torch.zeros_like(x)


class _TinyCfgModel(nn.Module):
    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        del t, kwargs
        bias = encoder_hidden_states.mean(dim=(1, 2)).view(x.shape[0], 1, 1, 1)
        return x + bias.expand_as(x)


def test_sample_loop_inference_amp_default_false():
    sig = inspect.signature(GaussianDiffusion.sample_loop)
    assert sig.parameters["inference_amp"].default is False
    flow_sig = inspect.signature(GaussianDiffusion._sample_loop_flow_matching)
    assert flow_sig.parameters["inference_amp"].default is False


def test_sample_loop_amp_off_cpu():
    diff = create_diffusion(num_timesteps=16)
    out = diff.sample_loop(
        _ConstVel(),
        (1, 4, 4, 4),
        device="cpu",
        dtype=torch.float32,
        num_inference_steps=2,
        cfg_scale=1.0,
        inference_amp=False,
    )
    assert out.shape == (1, 4, 4, 4)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float32


def test_flow_matching_loop_amp_off_cpu():
    diff = create_diffusion(num_timesteps=16)
    out = diff.sample_loop(
        _ConstVel(),
        (1, 4, 4, 4),
        device="cpu",
        dtype=torch.float32,
        num_inference_steps=3,
        cfg_scale=1.0,
        flow_matching_sample=True,
        inference_amp=False,
    )
    assert out.shape == (1, 4, 4, 4)
    assert torch.isfinite(out).all()


def test_cpu_inference_amp_true_is_noop():
    probe = _AutocastProbe()
    diff = create_diffusion(num_timesteps=16)
    diff.sample_loop(
        probe,
        (1, 4, 4, 4),
        device="cpu",
        num_inference_steps=1,
        cfg_scale=1.0,
        inference_amp=True,
    )
    assert probe.saw_autocast is False


def test_batched_cfg_amp_default_matches_sequential():
    model = _TinyCfgModel()
    x = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    mk_c = {"encoder_hidden_states": torch.randn(2, 3, 16)}
    mk_u = {"encoder_hidden_states": torch.randn(2, 3, 16)}
    batched = batched_cfg_forward(
        model,
        x,
        t,
        model_kwargs_cond=mk_c,
        model_kwargs_uncond=mk_u,
        cfg_scale=7.5,
        amp=False,
    )
    oc = model(x, t, **mk_c)
    ou = model(x, t, **mk_u)
    sequential = ou + 7.5 * (oc - ou)
    assert torch.allclose(batched, sequential, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inference AMP")
def test_cuda_inference_amp_enables_autocast():
    probe = _AutocastProbe().cuda()
    diff = create_diffusion(num_timesteps=16)
    out = diff.sample_loop(
        probe,
        (1, 4, 4, 4),
        device="cuda",
        num_inference_steps=2,
        cfg_scale=1.0,
        inference_amp=True,
        inference_amp_dtype=torch.bfloat16,
    )
    assert probe.saw_autocast is True
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inference AMP")
def test_cuda_flow_inference_amp_enables_autocast():
    probe = _AutocastProbe().cuda()
    diff = create_diffusion(num_timesteps=16)
    out = diff.sample_loop(
        probe,
        (1, 4, 4, 4),
        device="cuda",
        num_inference_steps=2,
        cfg_scale=1.0,
        flow_matching_sample=True,
        inference_amp=True,
        inference_amp_dtype=torch.bfloat16,
    )
    assert probe.saw_autocast is True
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inference AMP")
def test_cuda_batched_cfg_amp_wraps_model():
    class _Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.flags: list[bool] = []

        def forward(self, x, t, encoder_hidden_states=None, **kwargs):
            del t, kwargs
            self.flags.append(bool(torch.is_autocast_enabled()))
            bias = encoder_hidden_states.mean(dim=(1, 2)).view(x.shape[0], 1, 1, 1)
            return x + bias.expand_as(x)

    model = _Probe().cuda()
    x = torch.randn(2, 4, 8, 8, device="cuda")
    t = torch.zeros(2, dtype=torch.long, device="cuda")
    mk_c = {"encoder_hidden_states": torch.randn(2, 3, 16, device="cuda")}
    mk_u = {"encoder_hidden_states": torch.randn(2, 3, 16, device="cuda")}
    out = batched_cfg_forward(
        model,
        x,
        t,
        model_kwargs_cond=mk_c,
        model_kwargs_uncond=mk_u,
        cfg_scale=7.5,
        amp=True,
        autocast_dtype=torch.bfloat16,
    )
    assert model.flags and all(model.flags)
    assert out.dtype == torch.float32
    assert out.shape == x.shape
