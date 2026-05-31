from pathlib import Path

from utils.modeling.hf_scaffold import (
    CONFIG_ONLY_IGNORE,
    has_local_weights,
    resolve_entries,
    scaffold_registry,
)
from utils.modeling.model_paths import pretrained_catalog


def test_scaffold_registry_has_new_boost_models():
    names = {e.name for e in scaffold_registry()}
    for expected in (
        "Florence-2-base",
        "BLIP2-opt-2.7b",
        "Qwen2-VL-2B-Instruct",
        "Depth-Anything-V2-Small",
        "CLIP-ViT-H-14",
        "OWLv2-base-patch16-ensemble",
        "ControlNet-Canny",
        "SmolVLM-256M-Instruct",
        "OneAlign",
        "Florence-2-large",
        "ControlNet-Depth",
        "MetaCLIP-B16",
        "SmolVLM2-2B-Instruct",
        "GIT-base-coco",
        "GroundingDINO-SwinT",
        "ControlNet-SoftEdge",
        "MUSIQ",
        "EVA02-CLIP-L-14",
        "LLaVA-1.5-7B",
        "InternVL2-2B",
        "Qwen2.5-VL-3B-Instruct",
        "ControlNet-HED",
        "ZoeDepth",
        "NSFW-Detector",
        "LLaVA-v1.6-Mistral-7B",
        "PaliGemma2-3B",
        "DINOv3-ViT-S16",
        "GFPGAN",
        "Watermark-Detector",
        "UMT5-XXL",
    ):
        assert expected in names


def test_resolve_entries_by_role():
    vlms = resolve_entries(roles=["vlm"])
    assert vlms
    assert all(e.role == "vlm" for e in vlms)
    assert "Florence-2-base" in {e.name for e in vlms}


def test_config_only_ignore_blocks_weights():
    assert any("safetensors" in p for p in CONFIG_ONLY_IGNORE)


def test_has_local_weights(tmp_path: Path):
    assert has_local_weights(tmp_path) is False
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert has_local_weights(tmp_path) is False
    (tmp_path / "model.safetensors").write_text("x", encoding="utf-8")
    assert has_local_weights(tmp_path) is True


def test_pretrained_catalog_includes_new_rows():
    names = {str(r["name"]) for r in pretrained_catalog()}
    assert "Florence-2-base" in names
    assert "BLIP2-opt-2.7b" in names
    assert "Qwen2-VL-2B-Instruct" in names
    assert "Depth-Anything-V2-Small" in names
    assert "ControlNet-Canny" in names
