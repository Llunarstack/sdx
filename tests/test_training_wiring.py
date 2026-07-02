"""Verify scrape → enrich → control → train field wiring (image pipeline only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.manifest_utils import negative_caption_from_row, pick_training_manifest, read_manifest_rows


def test_pick_training_manifest_prefers_enriched(tmp_path: Path):
    combined = tmp_path / "combined" / "manifest.jsonl"
    enriched = tmp_path / "enriched" / "manifest.jsonl"
    combined.parent.mkdir(parents=True)
    enriched.parent.mkdir(parents=True)
    combined.write_text(json.dumps({"image_path": "a.png", "caption": "raw"}) + "\n", encoding="utf-8")
    enriched.write_text(
        json.dumps({"image_path": "a.png", "caption": "researched prompt", "tag_sources": ["creative_rag"]}) + "\n",
        encoding="utf-8",
    )
    assert pick_training_manifest(combined, enriched) == enriched


def test_pick_training_manifest_falls_back_to_combined(tmp_path: Path):
    combined = tmp_path / "combined" / "manifest.jsonl"
    enriched = tmp_path / "enriched" / "manifest.jsonl"
    combined.parent.mkdir(parents=True)
    enriched.parent.mkdir(parents=True)
    combined.write_text(json.dumps({"image_path": "a.png", "caption": "raw"}) + "\n", encoding="utf-8")
    enriched.write_text("", encoding="utf-8")
    assert pick_training_manifest(combined, enriched) == combined


def test_negative_caption_from_research_fields():
    row = {"negative_prompt_hint": "low quality, blurry"}
    assert negative_caption_from_row(row) == "low quality, blurry"
    row2 = {"negative_caption": "bad anatomy", "negative_prompt_hint": "ignored"}
    assert negative_caption_from_row(row2) == "bad anatomy"


def test_dataset_reads_negative_prompt_hint(tmp_path: Path):
    from data.t2i_dataset import Text2ImageDataset

    import numpy as np
    from PIL import Image

    root = tmp_path / "data"
    root.mkdir(parents=True)
    img = root / "img.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img)
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": str(img),
                "caption": "1girl, solo",
                "negative_prompt_hint": "worst quality, blurry",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ds = Text2ImageDataset(str(manifest), image_size=32)
    item = ds[0]
    assert "blurry" in item["negative_caption"].lower()


def test_control_manifest_row_shape(tmp_path: Path):
    """Control preprocess output must include fields train.py reads."""
    from setup.preprocess_control_maps import _extract_one

    data_root = tmp_path / "sdx"
    img_dir = data_root / "danbooru" / "images"
    img_dir.mkdir(parents=True)
    from PIL import Image
    import numpy as np

    img = img_dir / "abc.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img)
    row = {
        "image_path": "danbooru/images/abc.png",
        "caption": "researched diffusion prompt",
        "negative_caption": "blurry",
        "md5": "abc",
    }
    out = _extract_one(data_root, row, control_type="canny", controls_dir=data_root / "controls")
    assert out is not None
    assert out.get("control_image")
    assert out.get("control_type") == "canny"
    assert out.get("caption") == "researched diffusion prompt"
    assert out.get("negative_caption") == "blurry"


def test_pipeline_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "train.py",
        "setup/download_datasets.py",
        "setup/merge_manifests.py",
        "setup/enrich_manifest_captions.py",
        "setup/build_rag_corpus.py",
        "setup/preprocess_control_maps.py",
        "setup/build_artist_index.py",
        "scripts/run_pipeline.py",
        "runpod/download.sh",
        "runpod/train.sh",
        "runpod/sample.sh",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    assert not missing, f"missing pipeline files: {missing}"


def test_train_cli_has_manifest_arg():
    from training.train_cli_parser import build_train_arg_parser

    p = build_train_arg_parser()
    args = p.parse_args(["--manifest-jsonl", "x.jsonl", "--data-path", "/data"])
    assert args.manifest_jsonl == "x.jsonl"


def test_enriched_row_fields_survive_merge_research():
    from setup.enrich_manifest_captions import _merge_research_row
    from utils.caption.prompt_research import PromptResearchResult

    row = {
        "caption": "1girl, solo",
        "character_tags": ["miku"],
        "copyright_tags": ["vocaloid"],
    }
    researched = PromptResearchResult(
        diffusion_prompt="idol on stage, spotlight",
        negative_prompt="low quality",
        sources=["creative_rag"],
    )
    merged = _merge_research_row(row, researched)
    assert "miku" in merged["caption"].lower()
    assert merged["negative_caption"] == "low quality"
    assert merged["booru_caption"] == "1girl, solo"
