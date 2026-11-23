"""Tests for data/: T2I dataset, caption utils, manifests, reverse search.
Also holds what used to be: test_data_pipeline, test_t2i_dataset, test_caption_truncate, test_caption_utils, test_enrich_manifest, test_jsonl_caption_hygiene_native, test_reverse_search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from data.caption_utils import (
    ANTI_BLEND_NEGATIVE,
    ANTI_BLEND_POSITIVE,
    add_anti_blending_and_count,
    truncate_caption_at_comma_boundary,
)
from data.t2i_dataset import EnhancedT2IDataset, Text2ImageDataset
from PIL import Image
from setup.manifests import _merge_research_row, _row_needs_enrich
from utils.captioning import PromptResearchResult, ReverseHit, hit_meets_threshold, parse_source_url

# ==================================================================================================
# test_data_pipeline: Tests for the data pipeline: dataset loading, caption processing, collation,
# ==================================================================================================


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
        lines.append(
            json.dumps(
                {
                    "image_path": str(img_path),
                    "caption": f"a test image number {i}",
                    "negative_caption": "blurry, low quality",
                    "weight": 1.0,
                }
            )
        )
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

    def test_relative_paths_with_data_root(self, tmp_path):
        from data.t2i_dataset import Text2ImageDataset

        data_root = tmp_path / "sdx_data"
        site_dir = data_root / "danbooru" / "images"
        site_dir.mkdir(parents=True)
        img = site_dir / "abc.png"
        _make_image(img)
        manifest = data_root / "combined" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"image_path": "danbooru/images/abc.png", "caption": "1girl, solo"}) + "\n",
            encoding="utf-8",
        )
        ds = Text2ImageDataset(str(manifest), image_size=32, data_root=str(data_root))
        assert len(ds) == 1
        item = ds[0]
        assert item["pixel_values"].shape == (3, 32, 32)

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
        from utils.training.objectives import image_mask_to_patch_weights

        mask = torch.rand(2, 1, 32, 32)
        weights = image_mask_to_patch_weights(mask, num_patches_h=4, num_patches_w=4)
        assert weights.shape == (2, 16)

    def test_image_mask_to_patch_weights_range(self):
        from utils.training.objectives import image_mask_to_patch_weights

        mask = torch.rand(2, 1, 32, 32).clamp(0, 1)
        weights = image_mask_to_patch_weights(mask, num_patches_h=4, num_patches_w=4)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0 + 1e-5

    def test_image_mask_to_patch_weights_invalid_shape(self):
        from utils.training.objectives import image_mask_to_patch_weights

        with pytest.raises(ValueError):
            image_mask_to_patch_weights(torch.rand(2, 3, 32, 32), num_patches_h=4, num_patches_w=4)

    def test_foreground_attention_alignment_loss_shape(self):
        from utils.training.objectives import foreground_attention_alignment_loss

        B, H, N, L = 2, 4, 16, 32
        attn = torch.rand(B, H, N, L)
        mask = torch.rand(B, N).clamp(0, 1)
        loss = foreground_attention_alignment_loss(attn, mask)
        assert loss.ndim == 0  # scalar

    def test_token_coverage_loss_shape(self):
        from utils.training.objectives import token_coverage_loss

        B, H, N, L = 2, 4, 16, 32
        attn = torch.rand(B, H, N, L)
        loss = token_coverage_loss(attn, target_coverage=0.025)
        assert loss.ndim == 0

    def test_merge_hierarchical_captions(self):
        from utils.training.objectives import merge_hierarchical_captions

        result = merge_hierarchical_captions(
            "base caption",
            caption_global="global scene",
            caption_local="local detail",
        )
        assert "base caption" in result
        assert "global scene" in result
        assert "local detail" in result

    def test_foveated_random_crop_box_bounds(self):
        from utils.training.objectives import foveated_random_crop_box

        y0, x0, y1, x1 = foveated_random_crop_box(64, 64, crop_frac=0.5)
        assert 0 <= y0 < y1 <= 64
        assert 0 <= x0 < x1 <= 64


# ==================================================================================================
# test_t2i_dataset: Text2ImageDataset cropping: bucket sizes and pixel alignment of image, mask, control and init inputs.
# ==================================================================================================


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


# ==================================================================================================
# test_caption_truncate: Tests for comma-safe caption truncation.
# ==================================================================================================


def test_truncate_keeps_whole_tags():
    parts = [f"tag{i}" for i in range(20)]
    caption = ", ".join(parts)
    out = truncate_caption_at_comma_boundary(caption, 40)
    assert "," not in out.split(",")[-1][:1] or len(out) <= 40
    for piece in out.split(","):
        assert piece.strip() in {f"tag{i}" for i in range(20)}


def test_truncate_single_long_tag():
    long_tag = "x" * 100
    assert truncate_caption_at_comma_boundary(long_tag, 50) == long_tag[:50]


# ==================================================================================================
# test_caption_utils: Caption helpers: multi-person anti-blending detection.
# ==================================================================================================


def test_single_character_caption_gets_no_anti_blending():
    assert add_anti_blending_and_count("1girl, solo, smile", "blurry") == ("1girl, solo, smile", "blurry")
    assert add_anti_blending_and_count("1 person walking a dog", "") == ("1 person walking a dog", "")


def test_several_people_get_anti_blending():
    for caption in ("1girl, 1boy, park", "2girls, beach", "3 people at a table", "a crowd at night"):
        cap, neg = add_anti_blending_and_count(caption, "blurry")
        assert cap == f"{caption}, {ANTI_BLEND_POSITIVE}"
        assert neg == f"blurry, {ANTI_BLEND_NEGATIVE}"


# ==================================================================================================
# test_enrich_manifest: Tests for enrich_manifest_captions helpers.
# ==================================================================================================


def test_row_needs_enrich_prompt_research():
    assert _row_needs_enrich({"tag_sources": ["creative_rag"]}, use_prompt_research=True) is False
    assert _row_needs_enrich({"tag_sources": ["booru"]}, use_prompt_research=True) is True
    assert _row_needs_enrich({"scene_summary": "x"}, use_prompt_research=False) is False


def test_merge_research_row_preserves_booru_identity():
    row = {
        "caption": "1girl, solo, blue_hair",
        "character_tags": ["hatsune_miku"],
        "copyright_tags": ["vocaloid"],
        "artist_tags": ["artist_name"],
        "tag_sources": ["booru"],
    }
    researched = PromptResearchResult(
        diffusion_prompt="solo girl standing on stage, spotlight, detailed illustration",
        sources=["vlm_uncensored", "creative_rag"],
    )
    merged = _merge_research_row(row, researched)
    cap = merged["caption"].lower()
    assert "hatsune miku" in cap
    assert "vocaloid" in cap
    assert "artist name" in cap
    assert "stage" in cap
    assert merged["booru_caption"] == row["caption"]
    assert "creative_rag" in merged["tag_sources"]


# ==================================================================================================
# test_jsonl_caption_hygiene_native
# ==================================================================================================


def test_jsonl_caption_hygiene_normalizes_unicode(tmp_path: Path):
    # Includes zero-width and compatibility chars; should normalize + strip safely.
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(
        json.dumps({"caption": "a\u200bgirl, cafe", "negative_caption": " \ufeffbad  ,  text"}) + "\n",
        encoding="utf-8",
    )

    from sdx_native.text_hygiene import normalize_manifest_jsonl

    n = normalize_manifest_jsonl(inp=inp, out=out)
    assert n == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["caption"] == "agirl, cafe"
    assert row["negative_caption"] == "bad, text"


def test_null_negative_and_non_object_rows(tmp_path: Path):
    # A null negative_caption used to be written back as the string "None"; a JSON array row crashed the pass.
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"caption": "cat", "negative_caption": None, "negative_prompt": "blurry"}, [1, 2], {"caption": "dog"}]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    from sdx_native.text_hygiene import normalize_manifest_jsonl

    assert normalize_manifest_jsonl(inp=inp, out=out) == 2
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["negative_caption"] is None
    assert first["negative_prompt"] == "blurry"


def test_caption_normalization_is_idempotent_for_fullwidth_commas():
    from sdx_native.text_hygiene import normalize_caption_for_training

    once = normalize_caption_for_training("a，b, c")
    assert once == "a, b, c"
    assert normalize_caption_for_training(once) == once


# ==================================================================================================
# test_reverse_search: Tests for SauceNAO + TinEye reverse search helpers.
# ==================================================================================================


def test_parse_danbooru_url():
    site, pid = parse_source_url("https://danbooru.donmai.us/posts/1234567")
    assert site == "danbooru"
    assert pid == "1234567"


def test_parse_e621_url():
    site, pid = parse_source_url("https://e621.net/posts/987654")
    assert site == "e621"
    assert pid == "987654"


def test_parse_pixiv_artworks_url():
    site, pid = parse_source_url("https://www.pixiv.net/en/artworks/555")
    assert site == "pixiv"
    assert pid == "555"


def test_tineye_booru_hit_lower_threshold():
    hit = ReverseHit(similarity=55.0, site="danbooru", site_id="1", engine="tineye")
    assert hit_meets_threshold(hit, 75.0) is True


def test_saucenao_html_parser():
    html = """
    <div class="resultadosub"><div class="resulttable"><div>87.3%</div>
    <a href="https://danbooru.donmai.us/posts/12345">danbooru</a>
    Characters: hatsune_miku</div></div>
    """
    from utils.captioning import _hits_from_saucenao_html

    hits = _hits_from_saucenao_html(html)
    assert hits
    assert hits[0].site == "danbooru"
    assert hits[0].site_id == "12345"
    assert hits[0].similarity == 87.3


def test_saucenao_hit_needs_full_threshold():
    hit = ReverseHit(similarity=55.0, site="danbooru", site_id="1", engine="saucenao")
    assert hit_meets_threshold(hit, 75.0) is False


def test_fetch_booru_post_parses_danbooru_and_e621(monkeypatch):
    import requests
    from utils.captioning import fetch_booru_post

    payloads = {
        "danbooru": {"tag_string_character": "hatsune_miku", "tag_string_artist": "kei", "tag_string": "a b"},
        "e621": {"post": {"tags": {"character": ["renamon"], "general": ["fur", " "], "meta": "x"}}},
    }

    class _Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_get(url, **kwargs):
        return _Resp(payloads["danbooru" if "danbooru" in url else "e621"])

    monkeypatch.setattr(requests, "get", fake_get)
    db = fetch_booru_post("danbooru", "12")
    assert db.character_tags == ["hatsune_miku"] and db.artist_tags == ["kei"] and db.tag_string == "a b"
    e6 = fetch_booru_post("e621", "34")
    assert e6.character_tags == ["renamon"] and e6.general_tags == ["fur"] and e6.tag_string == "renamon fur"
    assert fetch_booru_post("pixiv", "1") is None
    assert fetch_booru_post("danbooru", "abc") is None
