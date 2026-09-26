"""End-to-end wiring tests for Invention Lab integration points."""

from __future__ import annotations

import argparse
import importlib
from unittest.mock import MagicMock

from PIL import Image
from pipelines.image_gen import ImageGenerateConfig, ImageGenerationPipeline
from utils.generation.inventions.registry import INVENTIONS, inventory
from utils.generation.inventions.wire import apply_invention_to_namespace, prepare_invention_runtime
from utils.generation.perfect_gen import build_sample_argv
from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_cli_passthrough import append_sample_repair_passthrough


def test_registry_has_100_and_paths_importable() -> None:
    assert len(INVENTIONS) == 100
    inv = inventory()
    assert inv["coded"] + inv.get("scaffold", 0) + inv.get("partial", 0) == 100
    seen: set[str] = set()
    for row in INVENTIONS:
        path = str(row.get("path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        importlib.import_module(path)


def test_prepare_runtime_and_namespace() -> None:
    rt = prepare_invention_runtime(
        "four red cubes left of blue spheres, no watermark",
        enable="auto",
        base_steps=28,
        adaptive_steps=True,
        auto_spectra=True,
    )
    assert rt.positive
    assert rt.steps is not None
    assert rt.invention_spectra is True

    args = argparse.Namespace(
        prompt="2girls holding hands, no hat",
        negative_prompt="",
        invention_stack="auto",
        invention_spectra=False,
        invention_auto_spectra=True,
        invention_adaptive_steps=True,
        steps=28,
        box_layout="",
        anti_bleed=False,
    )
    out = apply_invention_to_namespace(args, work_dir="")
    assert out is not None
    assert args.prompt
    assert args.invention_spectra is True
    assert int(args.steps) >= 12


def test_cli_parser_invention_flags() -> None:
    p = build_sample_parser()
    ns = p.parse_args(
        [
            "--ckpt",
            "x.pt",
            "--prompt",
            "a cat",
            "--invention-stack",
            "auto",
            "--invention-spectra",
            "--invention-auto-spectra",
            "--invention-adaptive-steps",
            "--invention-self-heal",
        ]
    )
    assert ns.invention_stack == "auto"
    assert ns.invention_spectra is True
    assert ns.invention_auto_spectra is True
    assert ns.invention_adaptive_steps is True
    assert ns.invention_self_heal is True


def test_passthrough_forwards_invention_flags() -> None:
    args = argparse.Namespace(
        invention_stack="auto",
        invention_spectra=True,
        invention_auto_spectra=True,
        invention_adaptive_steps=True,
        invention_self_heal=False,
        corpus_jsonl="data/train.jsonl",
        corpus_top_k=4,
        corpus_source="danbooru",
        corpus_root="",
        cast_json="cast.json",
    )
    cmd: list[str] = ["python", "sample.py"]
    append_sample_repair_passthrough(cmd, args)
    assert "--invention-stack" in cmd and "auto" in cmd
    assert "--invention-spectra" in cmd
    assert "--corpus-jsonl" in cmd
    assert "--cast-json" in cmd


def test_image_pipeline_applies_invention_stack(monkeypatch) -> None:
    captured: dict = {}

    def fake_sample_one_image_pil(**kwargs):
        captured.update(kwargs)
        return Image.new("RGB", (8, 8), color="white")

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
        config=ImageGenerateConfig(
            prompt="four red cubes left of blue spheres",
            steps=20,
            invention_stack="auto",
            invention_adaptive_steps=True,
            invention_auto_spectra=True,
            auto_layout=False,
        ),
    )
    out = pipe.generate()
    assert out.size == (8, 8)
    assert "invention_stack" not in captured
    assert captured["num_inference_steps"] >= 12
    assert captured.get("invention_spectra") is True
    assert "cube" in str(captured["prompt"]).lower() or "exactly" in str(captured["prompt"]).lower()


def test_perfect_gen_argv_includes_inventions() -> None:
    argv = build_sample_argv("portrait of a girl", invention_stack="auto")
    assert "--invention-stack" in argv
    assert "--invention-spectra" in argv


def test_models_exports_invention_arch() -> None:
    import models

    assert models.RelationTransformerHead is not None
    m = models.RelationTransformerHead(dim=32, n_relations=4, n_heads=2)
    import torch

    x = torch.randn(1, 3, 32)
    y = m(x)
    assert y.shape[-1] == 4
