"""Tests for ``utils.generation.latent_edit_helpers``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from utils.generation import latent_edit_helpers as le


def test_strength_to_start_timestep_clamp() -> None:
    assert le.strength_to_start_timestep(0.0, 1000) == 1
    assert le.strength_to_start_timestep(1.0, 1000) == 999
    assert le.strength_to_start_timestep(0.5, 100) == 50


def test_latent_hw_from_px() -> None:
    assert le.latent_hw_from_px(512) == 64


def test_pil_tensor_roundtrip() -> None:
    pil = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB")
    t = le.pil_rgb_to_tensor_m11(pil)
    assert t.shape == (1, 3, 16, 16)
    pil2 = le.tensor_m11_bchw_to_pil01(t)
    assert pil2.size == (16, 16)


def test_mask_pixel_to_latent() -> None:
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, :32, :] = 1.0
    ml = le.mask_pixel_to_latent(m, 8, 8)
    assert ml.shape == (1, 1, 8, 8)
    assert float(ml[0, 0, 0, 0].item()) == 1.0


def test_compose_outpaint_canvas() -> None:
    base = Image.new("RGB", (32, 32), color=(255, 0, 0))
    canvas, mask = le.compose_outpaint_canvas(base, 64, 64, anchor="center")
    assert canvas.size == (64, 64)
    assert mask.size == (64, 64)
    assert mask.getpixel((0, 0)) == 255
    assert mask.getpixel((32, 32)) == 0


def test_compose_outpaint_anchor_topleft() -> None:
    base = Image.new("RGB", (8, 8), color=(0, 255, 0))
    _, mask = le.compose_outpaint_canvas(base, 16, 16, anchor="topleft")
    assert mask.getpixel((4, 4)) == 0
    assert mask.getpixel((15, 15)) == 255


def test_build_img2img_initial_latent_vp() -> None:
    diffusion = MagicMock()
    diffusion.q_sample.side_effect = lambda x, t, noise=None: x + 0.1

    z0 = torch.ones(1, 4, 4, 4)
    x_init, t0 = le.build_img2img_initial_latent(diffusion, z0, num_timesteps=100, strength=0.3, use_flow_sample=False)
    assert t0 == 30
    assert x_init.shape == z0.shape
    diffusion.q_sample.assert_called_once()


def test_build_inpaint_legacy_vs_mdm() -> None:
    diffusion = MagicMock()

    def _q(x, t, noise=None):
        n = noise if noise is not None else torch.zeros_like(x)
        return x * 0.5 + n * 0.5

    diffusion.q_sample.side_effect = _q

    z0 = torch.ones(1, 4, 2, 2)
    m = torch.zeros_like(z0[:, :1, :, :])
    m[:, :, 1:, 1:] = 1.0

    leg = le.build_inpaint_initial_latent(diffusion, z0, m, num_timesteps=50, strength=0.4, inpaint_mode="legacy")
    assert leg.inpaint_freeze_known is False
    assert leg.x_init.shape == z0.shape

    mdm = le.build_inpaint_initial_latent(diffusion, z0, m, num_timesteps=50, strength=0.4, inpaint_mode="mdm")
    assert mdm.inpaint_freeze_known is True
    assert mdm.inpaint_mask is not None
    kw = mdm.sample_loop_kwargs()
    assert kw["inpaint_freeze_known"] is True
    assert "x_init" in kw


def test_prepare_latent_edit_from_paths_img2img(tmp_path: Path) -> None:
    img_path = tmp_path / "a.png"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(img_path)

    class _Dist:
        def sample(self):
            return torch.zeros(1, 4, 8, 8)

    vae = MagicMock()
    enc = MagicMock()
    enc.latent_dist = _Dist()
    vae.encode.return_value = enc

    diffusion = MagicMock()
    diffusion.q_sample.return_value = torch.full((1, 4, 8, 8), 0.25)

    out = le.prepare_latent_edit_from_paths(
        vae=vae,
        diffusion=diffusion,
        init_image_path=img_path,
        image_size_px=64,
        strength=0.2,
        num_timesteps=100,
        latent_scale=0.18215,
        ae_type="kl",
        rae_bridge=None,
        mask_path=None,
        use_flow_sample=False,
    )
    assert out.start_timestep == 20
    assert out.x_init.shape == (1, 4, 8, 8)
    assert out.inpaint_mask is None


def test_prepare_latent_edit_from_paths_inpaint_rejects_flow(tmp_path: Path) -> None:
    img_path = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(img_path)
    m_path = tmp_path / "m.png"
    Image.new("L", (32, 32), 255).save(m_path)

    vae = MagicMock()
    diffusion = MagicMock()
    with pytest.raises(ValueError, match="use_flow_sample"):
        le.prepare_latent_edit_from_paths(
            vae=vae,
            diffusion=diffusion,
            init_image_path=img_path,
            image_size_px=32,
            strength=0.5,
            num_timesteps=50,
            latent_scale=1.0,
            mask_path=m_path,
            use_flow_sample=True,
        )


def test_load_aux_rgb_tensor(tmp_path: Path) -> None:
    p = tmp_path / "c.png"
    Image.new("RGB", (20, 20), color=(1, 2, 3)).save(p)
    t = le.load_aux_rgb_tensor(p, 16)
    assert t.shape == (1, 3, 16, 16)


def test_blend_latents() -> None:
    a = torch.ones(1, 1, 2, 2)
    b = torch.zeros(1, 1, 2, 2)
    al = torch.full((1, 1, 2, 2), 0.25)
    out = le.blend_latents(a, b, al)
    assert out.shape == a.shape
    assert float(out[0, 0, 0, 0].item()) == pytest.approx(0.25)
