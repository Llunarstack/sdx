"""Tests for ``pipelines.book_comic.book_model_readiness`` and ViT peek helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


def test_peek_vit_quality_config(tmp_path: Path) -> None:
    from vit_quality.checkpoint_utils import peek_vit_quality_config

    p = tmp_path / "vq.pt"
    torch.save({"config": {"use_ar_conditioning": True, "image_size": 384}}, p)
    cfg = peek_vit_quality_config(p)
    assert cfg.get("use_ar_conditioning") is True
    assert int(cfg.get("image_size", 0)) == 384


def test_run_book_preflight_off() -> None:
    from pipelines.book_comic import book_model_readiness

    ns = SimpleNamespace(ckpt="__no_such_ckpt__", pick_vit_ckpt="", lora=[], control=[])
    e, w = book_model_readiness.run_book_preflight(ns, mode="off")
    assert e == [] and w == []


def test_run_book_preflight_strict_missing_ckpt(tmp_path: Path) -> None:
    from pipelines.book_comic import book_model_readiness

    ns = SimpleNamespace(
        ckpt=str(tmp_path / "nope.pt"),
        pick_vit_ckpt="",
        lora=[],
        control=[],
        control_image="",
        reference_image="",
        reference_adapter_pt="",
        tags_file="",
        character_sheet="",
        visual_memory="",
        consistency_json="",
    )
    errs, _ = book_model_readiness.run_book_preflight(ns, mode="strict", resolved_pick_best="clip")
    assert errs


def test_collect_dual_model_ar_mismatch() -> None:
    from pipelines.book_comic import book_model_readiness

    ns = SimpleNamespace(pick_vit_ckpt="x.pt", pick_vit_ar_blocks=4, pick_vit_ar_from_ckpt=False)
    w = book_model_readiness.collect_dual_model_alignment_warnings(
        ns,
        dit_ar_blocks=2,
        vit_cfg={"use_ar_conditioning": True, "image_size": 224},
    )
    assert any("num_ar_blocks" in x or "2" in x for x in w)


def test_book_model_stack_snapshot() -> None:
    from pipelines.book_comic import book_model_readiness

    ns = SimpleNamespace(
        ckpt="c.pt",
        pick_vit_ckpt="",
        lora=["a.safetensors:1"],
        control=[],
        pick_vit_ar_blocks=-1,
        pick_vit_ar_from_ckpt=False,
    )
    snap = book_model_readiness.book_model_stack_snapshot(ns, dit_ar_blocks=0, vit_cfg={})
    assert snap["dit_num_ar_blocks"] == 0
    assert snap["lora_count"] == 1
