"""utils.ideas: registry, the stack's hooks, every idea on real inputs, sampler wiring and the Rust kernels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from diffusion.gaussian_diffusion import create_diffusion
from utils.ideas import HOOKS, IDEAS, IdeaStack, ideas_table, kernels, parse_ideas

LATENT_HOOKS = ("noise", "scale", "delta", "x0", "step")


def _names(hook: str) -> list[str]:
    return [n for n, i in IDEAS.items() if i.hook == hook]


def _stack(name: str, strength: float = 1.0, n: int = 8, shape=(4, 4, 32, 32)) -> IdeaStack:
    s = IdeaStack(f"{name}={strength}", seed=3, native=16, prompt="a red fox, a blue lake, mountains and snow")
    s.begin(n, shape)
    return s


def _image(h: int = 48, w: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w, y / h, 0.5 + 0.4 * np.sin(x / 5.0)], -1)
    img[10:30, 20:40] = (0.95, 0.85, 0.80)
    img[35:40, 5:60] = 0.05
    return np.clip(img, 0, 1).astype(np.float32)


def test_registry_has_100_unique_ideas_on_every_hook():
    assert len(IDEAS) == 100
    assert {i.hook for i in IDEAS.values()} == set(HOOKS)
    assert all(i.summary.strip() and i.fn.__name__ == n for n, i in IDEAS.items())
    assert "100 ideas" in ideas_table() and all(f"[{h}]" in ideas_table() for h in HOOKS)


def test_parse_ideas_strengths_and_unknown_names():
    assert parse_ideas(" noise_pyramid, x0_soft_clip=0.5 ,") == [("noise_pyramid", 1.0), ("x0_soft_clip", 0.5)]
    with pytest.raises(ValueError, match="noise_pyramid"):
        parse_ideas("noise_pyramd")
    assert not IdeaStack("noise_pyramid=0")  # strength 0 is dropped, so the stack is a no-op
    assert IdeaStack("noise_pyramid,img_deband").names("image") == ["img_deband"]


@pytest.mark.parametrize("name", _names("noise"))
def test_noise_ideas_keep_unit_gaussian_statistics(name):
    torch.manual_seed(0)
    x = torch.randn(4, 4, 32, 32)
    out = _stack(name).noise(x)
    assert out.shape == x.shape and out.dtype == x.dtype and torch.isfinite(out).all()
    assert abs(float(out.std()) - 1.0) < 0.15 and abs(float(out.mean())) < 0.1
    assert not torch.equal(out, x) or name in ()  # every noise idea changes a batch-of-4 latent


@pytest.mark.parametrize("name", _names("scale"))
def test_scale_ideas_stay_reasonable(name):
    stack = _stack(name, shape=(1, 4, 64, 64))
    ws = []
    for i in range(8):
        stack.at(i)
        ws.append(stack.scale(7.0))
    assert all(1.0 < w < 12.0 for w in ws)
    assert stack.scale(1.0) == pytest.approx(1.0)  # CFG 1 (no guidance) stays no guidance


@pytest.mark.parametrize("name", _names("delta") + _names("x0") + _names("step"))
def test_latent_ideas_run_every_step(name):
    torch.manual_seed(1)
    stack = _stack(name)
    x_prev, x_new, d_cond, delta = (torch.randn(4, 4, 32, 32) for _ in range(4))
    for i in range(8):
        stack.at(i)
        hook = IDEAS[name].hook
        if hook == "delta":
            out = stack.delta(delta, d_cond, 5.0)
        elif hook == "x0":
            out = stack.x0(d_cond)
        else:
            out = stack.step(x_new, x_prev, d_cond)
        assert out.shape == x_new.shape and torch.isfinite(out).all(), (name, i)
    assert stack.step(x_new.half(), x_prev.half(), d_cond.half()).dtype == torch.float16  # dtype is preserved


@pytest.mark.parametrize("name", _names("image"))
def test_image_ideas_run_and_stay_in_range(name):
    img = _image()
    out = IDEAS[name].fn(img.copy(), _stack(name).items[0][1])
    assert out.shape == img.shape and np.isfinite(out).all()
    assert float(out.min()) > -0.25 and float(out.max()) < 1.25
    via_stack = IdeaStack(name, seed=1).image((img * 255).astype(np.uint8))
    assert via_stack.dtype == np.uint8 and via_stack.shape == img.shape


def test_semantics_of_key_ideas():
    torch.manual_seed(2)
    x = torch.randn(4, 4, 16, 16)
    q = _stack("noise_orthogonal_batch", shape=x.shape).noise(x).flatten(1)
    cos = torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(q, dim=1).T
    assert (cos - torch.eye(4)).abs().max() < 0.05
    sym = _stack("noise_mirror_h", shape=x.shape).noise(x)
    assert torch.allclose(sym, sym.flip(-1), atol=1e-5)
    centered = _stack("delta_channel_center").delta(x, x, 5.0)
    assert centered.mean((-2, -1)).abs().max() < 1e-5
    assert torch.allclose(_stack("x0_mirror_h").x0(x), _stack("x0_mirror_h").x0(x).flip(-1), atol=1e-5)
    img = _image()
    dith = IDEAS["img_blue_noise_dither"].fn(img, _stack("img_blue_noise_dither").items[0][1])
    assert np.allclose(dith * 255, np.round(dith * 255), atol=1e-3)
    assert abs(float(dith.mean() - img.mean())) < 1.0 / 255


def _cfg_model(x, t, encoder_hidden_states=None, **_):
    return 0.1 * x + 0.05 * encoder_hidden_states.mean((1, 2)).view(-1, 1, 1, 1)


class _Model(torch.nn.Module):
    def forward(self, x, t, **kw):
        return _cfg_model(x, t, **kw)


def _sample(flow: bool, ideas):
    torch.manual_seed(0)
    extra = dict(flow_matching_sample=True, flow_solver="euler") if flow else dict(solver="ddim", scheduler="ddim")
    cond = {"encoder_hidden_states": torch.ones(2, 3, 8)}
    uncond = {"encoder_hidden_states": torch.zeros(2, 3, 8)}
    return create_diffusion().sample_loop(
        _Model(),
        (2, 4, 16, 16),
        cfg_scale=4.0,
        model_kwargs_cond=cond,
        model_kwargs_uncond=uncond,
        num_inference_steps=6,
        device="cpu",
        ideas=ideas,
        **extra,
    )


@pytest.mark.parametrize("flow", [False, True])
def test_sample_loop_runs_one_idea_per_hook_and_is_exact_without_ideas(flow):
    base = _sample(flow, None)
    assert torch.equal(base, _sample(flow, IdeaStack("")))
    assert torch.equal(base, _sample(flow, IdeaStack("noise_pyramid=0,x0_soft_clip=0")))
    spec = "noise_shared_layout,cfg_bell,delta_channel_center,x0_temporal_ema,step_std_guard"
    out = _sample(flow, IdeaStack(spec, seed=1))
    assert torch.isfinite(out).all() and not torch.equal(out, base)


def test_rust_and_numpy_kernels_agree(monkeypatch):
    from sdx_native import ffi

    img = _image(24, 32)
    noisy = np.random.default_rng(0).random((24, 32, 3)).astype(np.float32)  # no equal-variance ties for Kuwahara
    if not ffi.rust_available():
        pytest.skip("Rust sdx_rs not built (cargo build --release in native/rust)")
    rust = {
        "bilateral": kernels.bilateral(img, 2, 1.5, 0.1),
        "median3": kernels.median3(img),
        "kuwahara": kernels.kuwahara(noisy, 2),
        "blue": kernels.blue_noise.__wrapped__(16, 1.5, 0),
    }
    for fn in ("bilateral", "median3", "kuwahara", "error_diffuse", "blue_noise"):
        monkeypatch.setattr(ffi, f"maybe_{fn}_rust", lambda *a, **k: None)
    assert np.abs(rust["bilateral"] - kernels.bilateral(img, 2, 1.5, 0.1)).max() < 1e-4
    assert np.array_equal(rust["median3"], kernels.median3(img))
    assert np.allclose(rust["kuwahara"], kernels.kuwahara(noisy, 2), atol=1e-6)
    ranks = np.sort((rust["blue"].ravel() * 256).astype(int))
    assert np.array_equal(ranks, np.arange(256))
    assert kernels.error_diffuse(img) is None  # NumPy has no Floyd-Steinberg; the idea falls back to blue noise


def test_numpy_blue_noise_is_a_rank_permutation_with_little_low_frequency_energy():
    bn = kernels._void_and_cluster(32, 1.5, 0)
    assert np.array_equal(np.sort((bn.ravel() * 1024).astype(int)), np.arange(1024))
    spec = np.abs(np.fft.fft2(bn - bn.mean())) ** 2
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(32), np.fft.fftfreq(32)))
    assert spec[(f > 0) & (f < 0.1)].mean() < 0.5 * spec[f > 0.3].mean()  # blue: energy sits at high frequency


def test_error_diffusion_keeps_the_mean(monkeypatch):
    from sdx_native import ffi

    if not ffi.rust_available():
        pytest.skip("Rust sdx_rs not built")
    flat = np.full((32, 32, 3), 100.4 / 255.0, np.float32)
    out = kernels.error_diffuse(flat)
    assert np.allclose(out * 255, np.round(out * 255), atol=1e-3)
    assert abs(float(out.mean() * 255) - 100.4) < 0.05


def test_bench_ideas_each_builds_one_variant_per_idea():
    from scripts.tools.bench import _idea_variants

    assert len(_idea_variants()) == 100
    image = _idea_variants("image")
    assert len(image) == 46 and image["img_deband"] == ["--ideas", "img_deband"]
    with pytest.raises(SystemExit):
        _idea_variants("bogus")


def test_horizon_level_straightens_a_tilted_horizon():
    import cv2

    img = np.full((120, 200, 3), 0.8, np.float32)
    img[60:, :] = 0.2
    m = cv2.getRotationMatrix2D((100, 60), 3.0, 1.0)
    tilted = cv2.warpAffine(img, m, (200, 120), borderMode=cv2.BORDER_REFLECT)

    def tilt(a):
        edges = cv2.Canny((a.mean(-1) * 255).astype(np.uint8), 60, 160)
        lines = cv2.HoughLines(edges, 1, np.pi / 720, 50)
        return abs(math.degrees(float(lines[0, 0, 1])) - 90.0)

    fixed = IDEAS["img_horizon_level"].fn(tilted, _stack("img_horizon_level").items[0][1])
    assert tilt(tilted) > 2.0 and tilt(fixed) < 1.0
