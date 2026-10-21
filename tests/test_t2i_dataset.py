"""Text2ImageDataset cropping: bucket sizes and pixel alignment of image, mask, control and init inputs."""

import json

import numpy as np
import pytest
import torch
from data.enhanced_dataset import EnhancedT2IDataset
from data.t2i_dataset import Text2ImageDataset
from PIL import Image


def _write_manifest(tmp_path, **row_extra):
    # Red channel = 40px checkerboard, also saved as the grounding mask, so alignment is checkable per pixel.
    yy, xx = np.mgrid[0:120, 0:200]
    checker = (((yy // 40) + (xx // 40)) % 2 * 255).astype(np.uint8)
    img = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    img[..., 0] = checker
    Image.fromarray(img).save(tmp_path / "img.png")
    Image.fromarray(checker).save(tmp_path / "mask.png")
    row = {"image_path": "img.png", "caption": "a barn at dusk", **row_extra}
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("crop_mode", ["center", "random", "largest_center"])
def test_non_square_bucket_crops_to_the_bucket_size(tmp_path, crop_mode):
    manifest = _write_manifest(tmp_path)  # landscape image into a portrait bucket
    ds = Text2ImageDataset(str(manifest), image_size=64, crop_mode=crop_mode, resolution_buckets=[(96, 64)])
    assert tuple(ds[0]["pixel_values"].shape) == (3, 96, 64)


def test_random_crop_keeps_mask_control_and_init_aligned(tmp_path):
    manifest = _write_manifest(tmp_path, grounding_mask="mask.png", control_image="img.png", init_image="img.png")
    ds = Text2ImageDataset(str(manifest), image_size=48, crop_mode="random")
    for _ in range(5):
        item = ds[0]
        assert torch.equal(item["control_image"], item["pixel_values"])
        assert torch.equal(item["init_pixel_values"], item["pixel_values"])
        # Mask and red channel go through the same resampling, so the binarised mask is exactly red > 50%.
        assert torch.equal(item["grounding_mask"][0].bool(), item["pixel_values"][0] > 0)


def test_enhanced_dataset_reads_its_manifest_fields(tmp_path):
    manifest = _write_manifest(tmp_path, spatial_layout=[[0.1, 0.1, 0.5, 0.5]], character_id=3)
    item = EnhancedT2IDataset(str(manifest), image_size=32)[0]
    assert item["spatial_layout"][0].tolist() == pytest.approx([0.1, 0.1, 0.5, 0.5])
    assert item["character_id"] == 3
