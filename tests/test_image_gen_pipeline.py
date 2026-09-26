"""Tests for ``pipelines.image_gen`` (mocked sampler; no GPU/T5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image
from pipelines.image_gen import ImageGenerateConfig, ImageGenerationPipeline


def test_generate_returns_sampled_pil(monkeypatch) -> None:
    expected = Image.new("RGB", (8, 8), color=(12, 34, 56))
    captured: dict = {}

    def fake_sample_one_image_pil(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "utils.generation.simple_latent_generate.sample_one_image_pil",
        fake_sample_one_image_pil,
    )

    pipe = ImageGenerationPipeline(
        model=MagicMock(),
        diffusion=MagicMock(),
        tokenizer=MagicMock(),
        text_encoder=MagicMock(),
        vae=MagicMock(),
        device="cpu",
        config=ImageGenerateConfig(prompt="a cat", steps=4, cfg_scale=1.5, seed=0),
    )
    out = pipe.generate()
    assert out is expected
    assert out.size == (8, 8)
    assert out.mode == "RGB"
    assert captured["prompt"] == "a cat"
    assert captured["num_inference_steps"] == 4
    assert captured["cfg_scale"] == 1.5
    assert captured["seed"] == 0


def test_generate_prompt_override(monkeypatch) -> None:
    expected = Image.new("RGB", (8, 8), color="white")

    def fake_sample_one_image_pil(**kwargs):
        fake_sample_one_image_pil.last_prompt = kwargs["prompt"]
        return expected

    monkeypatch.setattr(
        "utils.generation.simple_latent_generate.sample_one_image_pil",
        fake_sample_one_image_pil,
    )

    pipe = ImageGenerationPipeline(
        model=MagicMock(),
        diffusion=MagicMock(),
        tokenizer=MagicMock(),
        text_encoder=MagicMock(),
        vae=MagicMock(),
        device="cpu",
        config=ImageGenerateConfig(prompt="default"),
    )
    assert pipe.generate("override prompt") is expected
    assert fake_sample_one_image_pil.last_prompt == "override prompt"


def test_generate_left_of_keeps_steps_and_layout_text(monkeypatch) -> None:
    expected = Image.new("RGB", (8, 8), color="white")
    captured: dict = {}

    def fake_sample_one_image_pil(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "utils.generation.simple_latent_generate.sample_one_image_pil",
        fake_sample_one_image_pil,
    )

    pipe = ImageGenerationPipeline(
        model=MagicMock(),
        diffusion=MagicMock(),
        tokenizer=MagicMock(),
        text_encoder=MagicMock(),
        vae=MagicMock(),
        device="cpu",
        config=ImageGenerateConfig(prompt="a cat", steps=4),
    )
    out = pipe.generate("a red cube to the left of a blue cube", num_inference_steps=3)
    assert out is expected
    assert captured["num_inference_steps"] == 3
    got = str(captured["prompt"]).lower()
    assert "left" in got or "red_cube" in got or "blue_cube" in got
