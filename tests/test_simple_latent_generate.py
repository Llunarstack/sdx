"""Tests for ``utils.generation.simple_latent_generate``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from utils.generation.simple_latent_generate import sample_one_image_pil, tensor_bchw01_to_pil


def test_tensor_bchw01_to_pil_roundtrip_shape() -> None:
    t = torch.full((1, 3, 8, 8), 0.5)
    pil = tensor_bchw01_to_pil(t)
    assert pil.size == (8, 8)
    assert pil.mode == "RGB"


def test_tensor_bchw01_to_pil_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        tensor_bchw01_to_pil(torch.zeros(1, 1, 4, 4))


@patch("sample.encode_text")
def test_sample_one_image_pil_smoke(mock_encode: MagicMock) -> None:
    """Minimal stack: patched T5 encode, fake diffusion + VAE decode."""
    dev = torch.device("cpu")
    emb = torch.zeros(1, 2, 8, dtype=torch.float32)
    mock_encode.return_value = emb

    diffusion = MagicMock()
    diffusion.sample_loop.return_value = torch.zeros(1, 4, 4, 4)

    dec = MagicMock()
    dec.sample = torch.full((1, 3, 32, 32), 0.25)
    vae = MagicMock()
    vae.decode.return_value = dec

    model = MagicMock()
    tokenizer = MagicMock()
    text_encoder = MagicMock()

    pil = sample_one_image_pil(
        model=model,
        diffusion=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        device=dev,
        prompt="a cat",
        negative_prompt="",
        image_size=32,
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=0,
    )
    assert pil.size == (32, 32)
    assert pil.mode == "RGB"
    assert mock_encode.call_count == 2
    diffusion.sample_loop.assert_called_once()
