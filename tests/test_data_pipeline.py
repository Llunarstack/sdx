"""
Tests for the data pipeline: dataset loading, caption processing, collation,
bucket sampler, and part-aware training utilities.

Run with:
    pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    img.save(str(path))


def _make_folder_dataset(root: Path, n: int = 4) -> Path:
    """Create a minimal folder-mode dataset."""
    subdir = root / "subject_a"
    subdir.mkdir(parents=True)
    for i in range(n):
        img_path = subdir / f"img_{i:03d}.png"
        txt_path = subdir / f"img_{i:03d}.txt"
        _make_image(img_path)
        txt_path.write_text(f"a photo of subject {i}, high quality", encoding="utf-8")
    return root


def _make_jsonl_dataset(root: Path, n: int = 4) -> Path:
    """Create a minimal JSONL manifest dataset."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    lines = []
    for i in range(n):
        img_path = root / f"img_{i:03d}.png"
        _make_image(img_path)
        lines.append(json.dumps({
            "image_path": str(img_path),
            "caption": f"a test image number {i}",
            "negative_caption": "blurry, low quality",
            "weight": 1.0,
        }))
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Text2ImageDataset — folder mode
# ---------------------------------------------------------------------------

class TestText2ImageDatasetFolder:
    def test_loads_samples(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(root), image_size=32)
        assert len(ds) == 4

    def test_item_keys(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(root), image_size=32)
        item = ds[0]
        assert "pixel_values" in item
        assert "caption" in item

    def test_pixel_values_shape(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(root), image_size=32)
        item = ds[0]
        assert item["pixel_values"].shape == (3, 32, 32)

    def test_pixel_values_range(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(root), image_size=32)
        pv = ds[0]["pixel_values"]
        assert pv.min() >= -1.0 - 1e-5
        assert pv.max() <= 1.0 + 1e-5

    def test_caption_is_string(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(root), image_size=32)
        assert isinstance(ds[0]["caption"], str)
        assert len(ds[0]["caption"]) > 0

    @pytest.mark.parametrize("crop_mode", ["center", "random", "largest_center"])
    def test_crop_modes(self, tmp_path, crop_mode):
        from data.t2i_dataset import Text2ImageDataset
        root = _make_folder_dataset(tmp_path / f"data_{crop_mode}")
        ds = Text2ImageDataset(str(root), image_size=32, crop_mode=crop_mode)
        pv = ds[0]["pixel_values"]
        assert pv.shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Text2ImageDataset — JSONL mode
# ---------------------------------------------------------------------------

class TestText2ImageDatasetJSONL:
    def test_loads_samples(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        manifest = _make_jsonl_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(manifest), image_size=32)
        assert len(ds) == 4

    def test_negative_caption_present(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        manifest = _make_jsonl_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(manifest), image_size=32)
        item = ds[0]
        assert "negative_caption" in item

    def test_weight_present(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset
        manifest = _make_jsonl_dataset(tmp_path / "data")
        ds = Text2ImageDataset(str(manifest), image_size=32)
        assert "weight" in ds[0]

    def test_train_shortcomings_mitigation_auto(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset

        root = tmp_path / "data_sc"
        root.mkdir(parents=True)
        manifest = root / "manifest.jsonl"
        img_path = root / "img_000.png"
        _make_image(img_path)
        manifest.write_text(
            json.dumps(
                {
                    "image_path": str(img_path),
                    "caption": "portrait, woman sitting in sunlight",
                    "negative_caption": "blurry",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ds = Text2ImageDataset(
            str(manifest),
            image_size=32,
            train_shortcomings_mitigation="auto",
            train_shortcomings_2d=False,
        )
        item = ds[0]
        assert "floating" in item["negative_caption"].lower() or "contradictory" in item["negative_caption"].lower()
        assert len(item["caption"]) > len("portrait, woman sitting in sunlight")

    def test_train_art_guidance_auto(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset

        root = tmp_path / "data_art"
        root.mkdir(parents=True)
        manifest = root / "manifest.jsonl"
        img_path = root / "img_000.png"
        _make_image(img_path)
        manifest.write_text(
            json.dumps(
                {
                    "image_path": str(img_path),
                    "caption": "digital painting portrait of a woman",
                    "negative_caption": "blurry",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ds = Text2ImageDataset(
            str(manifest),
            image_size=32,
            train_art_guidance_mode="auto",
            train_art_guidance_photography=True,
            train_anatomy_guidance="lite",
        )
        item = ds[0]
        assert "blurry" in item["negative_caption"].lower()
        assert "bad anatomy" in item["negative_caption"].lower()
        assert len(item["caption"]) > len("digital painting portrait of a woman")

    def test_train_style_guidance_auto(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset

        root = tmp_path / "data_style"
        root.mkdir(parents=True)
        manifest = root / "manifest.jsonl"
        img_path = root / "img_000.png"
        _make_image(img_path)
        manifest.write_text(
            json.dumps(
                {
                    "image_path": str(img_path),
                    "caption": "anime manga hero, fortnite style",
                    "negative_caption": "blurry",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ds = Text2ImageDataset(
            str(manifest),
            image_size=32,
            train_style_guidance_mode="auto",
            train_style_guidance_artists=True,
        )
        item = ds[0]
        assert "blurry" in item["negative_caption"].lower()
        assert "style drift" in item["negative_caption"].lower() or "inconsistent" in item["negative_caption"].lower()
        assert len(item["caption"]) > len("anime manga hero, fortnite style")


# ---------------------------------------------------------------------------
# collate_t2i
# ---------------------------------------------------------------------------

class TestCollateT2I:
    def _make_batch(self, n: int = 3, size: int = 32):
        return [
            {
                "pixel_values": torch.randn(3, size, size),
                "caption": f"caption {i}",
                "negative_caption": "bad",
                "weight": 1.0,
                "difficulty": 0.5,
            }
            for i in range(n)
        ]

    def test_pixel_values_stacked(self):
        from data.t2i_dataset import collate_t2i
        batch = collate_t2i(self._make_batch(3))
        assert batch["pixel_values"].shape == (3, 3, 32, 32)

    def test_captions_list(self):
        from data.t2i_dataset import collate_t2i
        batch = collate_t2i(self._make_batch(3))
        assert isinstance(batch["captions"], list)
        assert len(batch["captions"]) == 3

    def test_sample_weights_tensor(self):
        from data.t2i_dataset import collate_t2i
        batch = collate_t2i(self._make_batch(3))
        assert "sample_weights" in batch
        assert batch["sample_weights"].shape == (3,)

    def test_grounding_mask_collation(self):
        """Batches with mixed grounding mask presence should pad missing masks."""
        from data.t2i_dataset import collate_t2i
        items = self._make_batch(3)
        items[0]["grounding_mask"] = torch.zeros(1, 32, 32)
        batch = collate_t2i(items)
        assert "grounding_mask" in batch
        assert batch["grounding_mask"].shape == (3, 1, 32, 32)
        assert "grounding_mask_valid" in batch
        assert batch["grounding_mask_valid"][0].item() is True
        assert batch["grounding_mask_valid"][1].item() is False


# ---------------------------------------------------------------------------
# Part-aware training utilities
# ---------------------------------------------------------------------------

class TestPartAwareTraining:
    def test_image_mask_to_patch_weights_shape(self):
        from utils.training.part_aware_training import image_mask_to_patch_weights
        mask = torch.rand(2, 1, 32, 32)
        weights = image_mask_to_patch_weights(mask, num_patches_h=4, num_patches_w=4)
        assert weights.shape == (2, 16)

    def test_image_mask_to_patch_weights_range(self):
        from utils.training.part_aware_training import image_mask_to_patch_weights
        mask = torch.rand(2, 1, 32, 32).clamp(0, 1)
        weights = image_mask_to_patch_weights(mask, num_patches_h=4, num_patches_w=4)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0 + 1e-5

    def test_image_mask_to_patch_weights_invalid_shape(self):
        from utils.training.part_aware_training import image_mask_to_patch_weights
        with pytest.raises(ValueError):
            image_mask_to_patch_weights(torch.rand(2, 3, 32, 32), num_patches_h=4, num_patches_w=4)

    def test_foreground_attention_alignment_loss_shape(self):
        from utils.training.part_aware_training import foreground_attention_alignment_loss
        B, H, N, L = 2, 4, 16, 32
        attn = torch.rand(B, H, N, L)
        mask = torch.rand(B, N).clamp(0, 1)
        loss = foreground_attention_alignment_loss(attn, mask)
        assert loss.ndim == 0  # scalar

    def test_token_coverage_loss_shape(self):
        from utils.training.part_aware_training import token_coverage_loss
        B, H, N, L = 2, 4, 16, 32
        attn = torch.rand(B, H, N, L)
        loss = token_coverage_loss(attn, target_coverage=0.025)
        assert loss.ndim == 0

    def test_merge_hierarchical_captions(self):
        from utils.training.part_aware_training import merge_hierarchical_captions
        result = merge_hierarchical_captions(
            "base caption",
            caption_global="global scene",
            caption_local="local detail",
        )
        assert "base caption" in result
        assert "global scene" in result
        assert "local detail" in result

    def test_foveated_random_crop_box_bounds(self):
        from utils.training.part_aware_training import foveated_random_crop_box
        y0, x0, y1, x1 = foveated_random_crop_box(64, 64, crop_frac=0.5)
        assert 0 <= y0 < y1 <= 64
        assert 0 <= x0 < x1 <= 64
