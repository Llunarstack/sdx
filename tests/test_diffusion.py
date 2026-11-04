"""Unit tests for diffusion preference DPO surrogate loss.

Also holds what used to be: test_diffusion_dpo_loss, test_diffusion_math, test_diffusion_regressions, test_flow_matching, test_pixel_perfect, test_diffusion_amp.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusion import create_diffusion
from diffusion import sampling as pp
from diffusion.flow_matching import flow_matching_per_sample_losses, sample_flow_time, shift_timestep_indices
from diffusion.gaussian_diffusion import GaussianDiffusion
from utils.generation.guidance import batched_cfg_forward
from utils.training.preference import safeguarded_dpo_preference_loss

# ==================================================================================================
# test_diffusion_dpo_loss: Unit tests for diffusion preference DPO surrogate loss.
# ==================================================================================================


def test_dpo_logit_clip_changes_loss_when_logits_would_be_large() -> None:
    win_p = torch.tensor(10.0)
    lose_p = torch.tensor(0.0)
    win_r = torch.tensor(5.0)
    lose_r = torch.tensor(5.0)
    unclamped = safeguarded_dpo_preference_loss(
        win_p, lose_p, win_r, lose_r, safeguard_strength=0.0, beta=500.0, logit_clip=None
    )
    clamped = safeguarded_dpo_preference_loss(
        win_p, lose_p, win_r, lose_r, safeguard_strength=0.0, beta=500.0, logit_clip=1.0
    )
    assert torch.isfinite(unclamped)
    assert torch.isfinite(clamped)
    assert abs(float(unclamped.detach()) - float(clamped.detach())) > 1e-6


def test_dpo_loss_grad_with_clip() -> None:
    win_p = torch.tensor(2.0, requires_grad=True)
    lose_p = torch.tensor(1.0, requires_grad=True)
    win_r = torch.tensor(1.5).detach()
    lose_r = torch.tensor(1.2).detach()
    loss = safeguarded_dpo_preference_loss(
        win_p, lose_p, win_r, lose_r, safeguard_strength=0.0, beta=100.0, logit_clip=5.0
    )
    loss.backward()
    assert win_p.grad is not None and torch.isfinite(win_p.grad)


# ==================================================================================================
# test_diffusion_math: Tests for diffusion math: schedules, SNR, alpha_cumprod, loss weights.
#
# These are the numerically critical paths — wrong values here silently corrupt
# training without any obvious error. Run with:
#
#     pytest tests/test_diffusion.py -v
# ==================================================================================================

# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------


class TestBetaSchedules:
    def test_linear_length(self):
        from diffusion.schedules import linear_beta_schedule

        b = linear_beta_schedule(1000)
        assert b.shape == (1000,)

    def test_linear_monotone_increasing(self):
        from diffusion.schedules import linear_beta_schedule

        b = linear_beta_schedule(1000)
        assert np.all(np.diff(b) >= 0), "linear betas must be non-decreasing"

    def test_linear_range(self):
        from diffusion.schedules import linear_beta_schedule

        b = linear_beta_schedule(1000)
        assert b[0] >= 1e-5
        assert b[-1] <= 0.999

    def test_cosine_length(self):
        from diffusion.schedules import cosine_beta_schedule

        b = cosine_beta_schedule(1000)
        assert b.shape == (1000,)

    def test_cosine_range(self):
        from diffusion.schedules import cosine_beta_schedule

        b = cosine_beta_schedule(1000)
        assert np.all(b > 0)
        assert np.all(b < 1)

    def test_squaredcos_v2_length(self):
        from diffusion.schedules import squared_cosine_beta_schedule_v2

        b = squared_cosine_beta_schedule_v2(1000)
        assert b.shape == (1000,)

    def test_squaredcos_v2_range(self):
        from diffusion.schedules import squared_cosine_beta_schedule_v2

        b = squared_cosine_beta_schedule_v2(1000)
        assert np.all(b > 0)
        assert np.all(b <= 0.999)

    def test_get_beta_schedule_dispatch(self):
        from diffusion.schedules import get_beta_schedule

        for name in ("linear", "cosine", "squaredcos_cap_v2"):
            b = get_beta_schedule(name, 100)
            assert b.shape == (100,), f"failed for {name}"

    def test_get_beta_schedule_unknown_raises(self):
        from diffusion.schedules import get_beta_schedule

        with pytest.raises(ValueError, match="Unknown beta schedule"):
            get_beta_schedule("bogus_schedule", 100)

    def test_sigmoid_length_and_range(self):
        from diffusion.schedules import sigmoid_beta_schedule

        b = sigmoid_beta_schedule(500)
        assert b.shape == (500,)
        assert np.all(b > 0) and np.all(b < 1)


# ---------------------------------------------------------------------------
# SNR / alpha_cumprod
# ---------------------------------------------------------------------------


class TestSnrUtils:
    def test_alpha_cumprod_monotone_decreasing(self):
        from diffusion.schedules import alpha_cumprod_from_betas, linear_beta_schedule

        betas = linear_beta_schedule(1000)
        ac = alpha_cumprod_from_betas(betas)
        assert ac.shape == (1000,)
        assert np.all(np.diff(ac) < 0), "alpha_cumprod must be strictly decreasing"

    def test_alpha_cumprod_bounds(self):
        from diffusion.schedules import alpha_cumprod_from_betas, linear_beta_schedule

        betas = linear_beta_schedule(1000)
        ac = alpha_cumprod_from_betas(betas)
        assert ac[0] < 1.0
        assert ac[-1] > 0.0

    def test_snr_positive(self):
        from diffusion.schedules import linear_beta_schedule, snr_from_betas

        betas = linear_beta_schedule(1000)
        snr = snr_from_betas(betas)
        assert np.all(snr >= 0)

    def test_snr_monotone_decreasing(self):
        from diffusion.schedules import linear_beta_schedule, snr_from_betas

        betas = linear_beta_schedule(1000)
        snr = snr_from_betas(betas)
        assert np.all(np.diff(snr) < 0), "SNR must decrease as noise increases"

    def test_snr_from_alpha_cumprod_matches_manual(self):
        from diffusion.schedules import alpha_cumprod_from_betas, linear_beta_schedule, snr_from_alpha_cumprod

        betas = linear_beta_schedule(100)
        ac = alpha_cumprod_from_betas(betas)
        snr = snr_from_alpha_cumprod(ac)
        expected = ac / (1.0 - ac + 1e-8)
        np.testing.assert_allclose(snr, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# GaussianDiffusion construction
# ---------------------------------------------------------------------------


class TestGaussianDiffusion:
    def test_create_diffusion_default(self):
        from diffusion import create_diffusion

        d = create_diffusion(num_timesteps=1000)
        assert d.num_timesteps == 1000
        assert d.alpha_cumprod.shape == (1000,)
        assert d.snr.shape == (1000,)

    def test_alpha_cumprod_tensor_monotone(self):
        from diffusion import create_diffusion

        d = create_diffusion(num_timesteps=1000)
        ac = d.alpha_cumprod.numpy()
        # Non-increasing; tail may be flat zero under float32.
        assert np.all(np.diff(ac) <= 0)
        assert ac[0] > ac[-1]

    def test_sqrt_tensors_positive(self):
        from diffusion import create_diffusion

        d = create_diffusion(num_timesteps=1000)
        assert (d.sqrt_alpha_cumprod >= 0).all()
        assert (d.sqrt_one_minus_alpha_cumprod >= 0).all()

    def test_q_sample_shape(self):
        from diffusion import create_diffusion

        d = create_diffusion(num_timesteps=1000)
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 1000, (2,))
        xt = d.q_sample(x, t)
        assert xt.shape == x.shape

    def test_offset_noise_per_channel(self):
        """Noise offset adds one constant per (sample, channel)."""
        from diffusion.gaussian_diffusion import offset_noise

        n = torch.zeros(2, 4, 8, 8)
        out = offset_noise(n, 0.1)
        assert out.shape == n.shape
        assert torch.allclose(out, out[..., :1, :1].expand_as(out))

    def test_prediction_types(self):
        from diffusion import create_diffusion

        for pt in ("epsilon", "v", "x0"):
            d = create_diffusion(num_timesteps=100, prediction_type=pt)
            assert d.prediction_type == pt

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "squaredcos_cap_v2"])
    def test_beta_schedules(self, schedule):
        from diffusion import create_diffusion

        d = create_diffusion(num_timesteps=100, beta_schedule=schedule)
        assert d.num_timesteps == 100

    def test_alpha_cumprod_matches_cumprod_one_minus_beta(self):
        """``GaussianDiffusion`` must agree with Π(1−βₜ) regardless of Rust/NumPy path."""
        from diffusion import create_diffusion
        from diffusion.schedules import get_beta_schedule

        for schedule in ("linear", "cosine", "squaredcos_cap_v2"):
            for n in (100, 1000):
                beta = get_beta_schedule(schedule, n)
                expected = np.cumprod(1.0 - np.asarray(beta, dtype=np.float64)).astype(np.float32)
                d = create_diffusion(num_timesteps=n, beta_schedule=schedule)
                np.testing.assert_allclose(
                    d.alpha_cumprod.numpy(),
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"alpha_cumprod mismatch for schedule={schedule!r} T={n}",
                )


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------


class TestLossWeights:
    def _make_tensors(self, T=100, device="cpu"):
        from diffusion.schedules import alpha_cumprod_from_betas, linear_beta_schedule, snr_from_alpha_cumprod

        betas = linear_beta_schedule(T)
        ac = torch.from_numpy(alpha_cumprod_from_betas(betas)).float().to(device)
        snr = torch.from_numpy(snr_from_alpha_cumprod(ac.numpy())).float().to(device)
        # Sample a batch of random timesteps
        t = torch.randint(0, T, (8,))
        return ac[t], snr[t]

    def test_min_snr_shape(self):
        from diffusion.schedules import get_timestep_loss_weight

        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight(
            "min_snr", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5
        )
        assert w.shape == (8,)
        assert (w > 0).all()

    def test_min_snr_capped(self):
        """min_snr weights must be <= 1 when gamma=5 (cap at SNR=5)."""
        from diffusion.schedules import get_timestep_loss_weight

        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight(
            "min_snr", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5
        )
        assert (w <= 1.0 + 1e-5).all(), "min_snr weights should be <= 1"

    def test_unit_weight_ones(self):
        from diffusion.schedules import get_timestep_loss_weight

        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight(
            "unit", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5
        )
        torch.testing.assert_close(w, torch.ones_like(w))

    def test_min_snr_soft_shape(self):
        from diffusion.schedules import get_timestep_loss_weight

        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight(
            "min_snr_soft", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5
        )
        assert w.shape == (8,)
        assert (w > 0).all()

    def test_edm_weight_positive(self):
        from diffusion.schedules import get_timestep_loss_weight

        ac, snr = self._make_tensors()
        w = get_timestep_loss_weight("edm", snr=snr, alpha_cumprod=ac, min_snr_gamma=5.0, loss_weighting_sigma_data=0.5)
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------


class TestTimestepSampling:
    def test_uniform_range(self):
        from diffusion.schedules import sample_training_timesteps

        t = sample_training_timesteps(256, 1000, device=torch.device("cpu"), mode="uniform")
        assert t.shape == (256,)
        assert t.min() >= 0
        assert t.max() < 1000

    def test_logit_normal_range(self):
        from diffusion.schedules import sample_training_timesteps

        t = sample_training_timesteps(256, 1000, device=torch.device("cpu"), mode="logit_normal")
        assert t.min() >= 0
        assert t.max() < 1000

    def test_high_noise_bias(self):
        """high_noise mode should have mean > 500 (biased toward high t)."""
        from diffusion.schedules import sample_training_timesteps

        t = sample_training_timesteps(2048, 1000, device=torch.device("cpu"), mode="high_noise")
        assert t.float().mean() > 500, "high_noise should bias toward large t"

    def test_low_noise_bias(self):
        """low_noise mode should have mean < 500 (biased toward small t)."""
        from diffusion.schedules import sample_training_timesteps

        t = sample_training_timesteps(2048, 1000, device=torch.device("cpu"), mode="low_noise")
        assert t.float().mean() < 500, "low_noise should bias toward small t"

    def test_unknown_mode_raises(self):
        from diffusion.schedules import sample_training_timesteps

        with pytest.raises(ValueError, match="Unknown timestep_sample_mode"):
            sample_training_timesteps(4, 1000, device=torch.device("cpu"), mode="bogus")

    def test_invalid_num_timesteps_raises(self):
        from diffusion.schedules import sample_training_timesteps

        with pytest.raises(ValueError):
            sample_training_timesteps(4, 0, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------


class TestFlowMatching:
    def test_velocity_target_shape(self):
        """flow_matching_per_sample_losses returns (B,) per-sample losses."""
        from diffusion.flow_matching import flow_matching_per_sample_losses

        class _IdentityModel(torch.nn.Module):
            def forward(self, x, t, **kw):
                return torch.zeros_like(x)

        model = _IdentityModel()
        x0 = torch.randn(4, 4, 8, 8)
        eps = torch.randn(4, 4, 8, 8)
        losses = flow_matching_per_sample_losses(model, x0, eps, num_timesteps=100, model_kwargs={})
        assert losses.shape == (4,)
        assert (losses >= 0).all()

    def test_shape_mismatch_raises(self):
        from diffusion.flow_matching import flow_matching_per_sample_losses

        class _M(torch.nn.Module):
            def forward(self, x, t, **kw):
                return x

        with pytest.raises(ValueError, match="must match shape"):
            flow_matching_per_sample_losses(
                _M(),
                torch.randn(2, 4, 8, 8),
                torch.randn(2, 4, 4, 4),
                num_timesteps=100,
                model_kwargs={},
            )


# ==================================================================================================
# test_diffusion_regressions: Regression tests for bugs fixed in the sampler, train.py and the root demo script.
# ==================================================================================================


class _ClassConditionedModel(torch.nn.Module):
    """Mimics ``models.dit.DiT``: ``forward(x, t, y)`` with no ``block_cache`` keyword."""

    def forward(self, x, t, y):
        return 0.1 * x


def test_sample_loop_without_cfg_does_not_pass_block_cache() -> None:
    out = create_diffusion(num_timesteps=50).sample_loop(
        _ClassConditionedModel(),
        (1, 4, 8, 8),
        model_kwargs_cond={"y": torch.tensor([3])},
        cfg_scale=1.0,
        num_inference_steps=3,
        device="cpu",
    )
    assert out.shape == (1, 4, 8, 8)


def test_demo_runs_sample_py_from_repo_root(monkeypatch) -> None:
    import demo

    seen: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda cmd: seen.append(cmd) or SimpleNamespace(returncode=0))
    args = SimpleNamespace(prompt="a cat", steps=2, cfg=5.0, out="o.png", preset="auto")
    demo._run_text_conditioned(Path("ckpt.pt"), args)
    assert seen[0][0] == sys.executable
    assert Path(seen[0][1]) == Path(demo.__file__).resolve().parent / "sample.py"


class _DtypeStrictTextModel(torch.nn.Module):
    """Like a real text projection, rejects text states whose dtype differs from the latent's."""

    def forward(self, x, t, encoder_hidden_states, **kwargs):
        assert encoder_hidden_states.dtype == x.dtype, (encoder_hidden_states.dtype, x.dtype)
        return 0.1 * x


def test_train_sample_preview_encodes_prompt_in_sampling_dtype(monkeypatch) -> None:
    from training import loop

    monkeypatch.setattr(
        loop, "encode_text", lambda caps, *a, dtype=torch.float32, **kw: torch.zeros(len(caps), 4, 8, dtype=dtype)
    )
    logged: list[tuple] = []
    tb_writer = SimpleNamespace(add_image=lambda tag, img, step, dataformats: logged.append(img.shape))
    vae = SimpleNamespace(decode=lambda x: SimpleNamespace(sample=x[:, :3]))
    cfg = SimpleNamespace(image_size=64, num_timesteps=50)
    diffusion = create_diffusion(num_timesteps=50)
    loop.log_sample_image(_DtypeStrictTextModel(), diffusion, None, None, vae, "cpu", cfg, 1, tb_writer)
    assert logged == [(8, 8, 3)]


# ==================================================================================================
# test_flow_matching: Rectified-flow training objective: time sampling, resolution shift, per-sample loss (image and video).
# ==================================================================================================


class _TinyDenoiser(nn.Module):
    """Maps (x_t, t) -> same-shaped velocity. Rank-agnostic (2D or video)."""

    def __init__(self, ch: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(ch))

    def forward(self, x, t, **kw):
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.scale.view(*shape)


def test_logit_normal_concentrates_in_the_middle():
    torch.manual_seed(0)
    s = sample_flow_time(20000, logit_normal=True, ln_mean=0.0, ln_std=1.0)
    assert 0.45 < float(s.mean()) < 0.55  # centered on 0.5
    mid_frac = float(((s > 0.25) & (s < 0.75)).float().mean())
    # Uniform would give 0.5 in that band; logit-normal puts more mass there.
    assert mid_frac > 0.55


def test_uniform_mode_matches_uniform():
    torch.manual_seed(0)
    s = sample_flow_time(20000, logit_normal=False)
    assert 0.47 < float(s.mean()) < 0.53
    mid_frac = float(((s > 0.25) & (s < 0.75)).float().mean())
    assert 0.47 < mid_frac < 0.53  # ~0.5, i.e. flat


def test_resolution_shift_pushes_toward_noise():
    torch.manual_seed(0)
    base = sample_flow_time(20000, logit_normal=True, shift=1.0)
    shifted = sample_flow_time(20000, logit_normal=True, shift=3.0)
    assert float(shifted.mean()) > float(base.mean())  # shift>1 raises s


def test_flow_loss_2d():
    torch.manual_seed(0)
    model = _TinyDenoiser(4)
    x0 = torch.randn(3, 4, 8, 8)
    eps = torch.randn(3, 4, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {}, logit_normal=True)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert model.scale.grad is not None


def test_flow_loss_video_shape():
    # Same objective must work for 5D video latents (B, C, T, H, W).
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 5, 8, 8)
    eps = torch.randn(2, 4, 5, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {}, logit_normal=True, shift=3.0)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_uniform_default_works():
    model = _TinyDenoiser(4)
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn(2, 4, 8, 8)
    loss = flow_matching_per_sample_losses(model, x0, eps, 1000, {})
    assert loss.shape == (2,)


def test_shift_timestep_indices_matches_sampled_shift():
    t = torch.tensor([0, 250, 500, 750, 999])
    assert torch.equal(shift_timestep_indices(t, 1000, 1.0), t)
    shifted = shift_timestep_indices(t, 1000, 3.0)
    assert shifted[0] == 0 and shifted[-1] == 999
    assert (shifted[1:-1] > t[1:-1]).all()  # shift > 1 moves toward noise


def test_given_t_idx_is_used():
    class _Probe(torch.nn.Module):
        def forward(self, x, t, **kw):
            self.seen = t
            return torch.zeros_like(x)

    model = _Probe()
    t = torch.tensor([3, 600])
    flow_matching_per_sample_losses(model, torch.zeros(2, 4, 4, 4), torch.ones(2, 4, 4, 4), 1000, {}, t_idx=t)
    assert torch.equal(model.seen, t)


# ==================================================================================================
# test_pixel_perfect: Tests for ``diffusion.sampling`` pixel-perfect canvas helpers.
# ==================================================================================================


def test_snap_to_multiple() -> None:
    assert pp.snap_to_multiple(100, 8, mode="floor") == 96
    assert pp.snap_to_multiple(100, 8, mode="ceil") == 104
    assert pp.snap_to_multiple(100, 8, mode="nearest") == 96


def test_latent_hw_roundtrip() -> None:
    assert pp.latent_hw_from_pixels(512, 512) == (64, 64)
    assert pp.pixels_from_latent_hw(64, 64) == (512, 512)


def test_pixel_stride_no_model() -> None:
    assert pp.pixel_stride_for_pipeline() == pp.LATENT_TO_PIXEL


def test_dit_rgb_stride_mock_model() -> None:
    class Emb:
        patch_size = 2

    class M:
        x_embedder = Emb()

    assert pp.dit_rgb_stride_px(M()) == 16


def test_resolve_with_dit_model_snaps_to_16() -> None:
    class Emb:
        patch_size = 2

    class M:
        x_embedder = Emb()

    h, w, spec = pp.resolve_pixel_perfect_hw(513, 500, model=M(), mode="nearest")
    assert h % 16 == 0 and w % 16 == 0
    assert spec.stride_px == 16
    assert spec.aligned_to_dit_patch is True
    assert spec.latent_h == h // 8


def test_resolve_square() -> None:
    h, w, spec = pp.resolve_pixel_perfect_hw(500, 510, square=True, mode="nearest")
    assert h == w
    assert h % 8 == 0


def test_validate_pixels_against_dit_ok() -> None:
    class Emb:
        img_size = 64
        patch_size = 2

    class M:
        x_embedder = Emb()

    pp.validate_pixels_against_dit(512, 512, M())
    pp.validate_pixels_against_dit(384, 688, M())


def test_validate_pixels_against_dit_bad() -> None:
    class Emb:
        img_size = 64
        patch_size = 2

    class M:
        x_embedder = Emb()

    with pytest.raises(ValueError, match="latent"):
        pp.validate_pixels_against_dit(504, 512, M())


def test_tag_manifest_pixel_perfect() -> None:
    h, w, spec = pp.resolve_pixel_perfect_hw(256, 256, mode="nearest")
    row = pp.tag_manifest_pixel_perfect({"caption": "x"}, spec)
    assert row["height_px"] == h
    assert row["pixel_perfect_stride_px"] == spec.stride_px


# ==================================================================================================
# test_diffusion_amp: Inference AMP kwargs and sampling-loop CPU smoke tests.
# ==================================================================================================


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
