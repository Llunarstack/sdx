"""
Tests for the ViT evaluator's semantic text featurizer (vit_quality/text_encoder.py)
and its wiring into the manifest dataset.

Covers the two things that must hold for the upgrade to be safe:
  1. The CLIP featurizer produces real, normalized, prompt-dependent embeddings.
  2. Every default path stays back-compatible with the legacy 8-D handcrafted
     vector, so pre-existing checkpoints load and score unchanged.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from PIL import Image
from vit_quality.dataset import ViTManifestDataset, text_feature_vector
from vit_quality.text_encoder import (
    HandcraftedTextFeaturizer,
    build_text_featurizer,
    featurizer_from_config,
)

CLIP_ID = "openai/clip-vit-base-patch32"


def _has_clip() -> bool:
    try:
        build_text_featurizer("clip", CLIP_ID, "cpu")
        return True
    except Exception:
        return False


def test_handcrafted_matches_legacy_vector():
    f = build_text_featurizer("handcrafted")
    assert isinstance(f, HandcraftedTextFeaturizer)
    assert f.dim == 8
    assert torch.equal(f("a photo of a cat"), text_feature_vector("a photo of a cat"))


def test_config_defaults_to_handcrafted():
    # A pre-upgrade checkpoint config has no text_embed_mode key.
    f = featurizer_from_config({})
    assert f.dim == 8
    # An explicit legacy dim also resolves to handcrafted.
    assert featurizer_from_config({"text_feat_dim": 8}).dim == 8


def test_clip_featurizer_semantics():
    if not _has_clip():
        import pytest

        pytest.skip("CLIP weights unavailable")
    f = build_text_featurizer("clip", CLIP_ID, "cpu")
    assert f.dim == 512
    e_apple = f("a red apple on a wooden table")
    e_apple2 = f("a red apple on a wooden table")
    e_car = f("a blue sports car on a highway")
    # L2-normalized
    assert abs(float(e_apple.norm()) - 1.0) < 1e-3
    # Deterministic + memoized
    assert torch.equal(e_apple, e_apple2)
    # Actually sees the prompt: different captions -> different embeddings
    assert float((e_apple - e_car).abs().sum()) > 1.0
    # embed_many matches single-call embeddings
    batch = f.embed_many(["a red apple on a wooden table", "a blue sports car on a highway"])
    assert batch.shape == (2, 512)
    assert torch.allclose(batch[0], e_apple, atol=1e-5)


def _make_manifest(tmp: Path, n: int = 4) -> str:
    img_dir = tmp / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        p = img_dir / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 30 % 256, 40, 200)).save(p)
        rows.append({"image_path": str(p), "caption": f"caption number {i}", "quality_label": float(i % 2)})
    mani = tmp / "m.jsonl"
    mani.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(mani)


def test_dataset_default_is_handcrafted():
    with tempfile.TemporaryDirectory() as td:
        mani = _make_manifest(Path(td))
        ds = ViTManifestDataset(mani, image_size=64)
        assert ds.text_feat_dim == 8
        assert ds[0]["text_features"].shape == (8,)


def test_dataset_with_clip_featurizer():
    if not _has_clip():
        import pytest

        pytest.skip("CLIP weights unavailable")
    with tempfile.TemporaryDirectory() as td:
        mani = _make_manifest(Path(td))
        f = build_text_featurizer("clip", CLIP_ID, "cpu")
        ds = ViTManifestDataset(mani, image_size=64, text_featurizer=f)
        assert ds.text_feat_dim == 512
        item = ds[0]
        assert item["text_features"].shape == (512,)
        # Precomputed features are used (not the 8-D fallback).
        assert item["text_features"].dtype == torch.float32


if __name__ == "__main__":
    test_handcrafted_matches_legacy_vector()
    print("[ok] handcrafted matches legacy")
    test_config_defaults_to_handcrafted()
    print("[ok] config defaults to handcrafted (back-compat)")
    test_dataset_default_is_handcrafted()
    print("[ok] dataset default handcrafted")
    if _has_clip():
        test_clip_featurizer_semantics()
        print("[ok] clip semantics")
        test_dataset_with_clip_featurizer()
        print("[ok] dataset + clip featurizer")
    else:
        print("[skip] CLIP weights unavailable")
    print("\nViT text-encoder upgrade verified.")
