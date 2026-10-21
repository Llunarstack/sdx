"""

Also holds what used to be: test_hf_control, test_hf_loaders, test_hf_reward, test_hf_scaffold.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
from utils.modeling.hf_control import (
    CONTROLNET_REGISTRY,
    extract_pil_proxy,
    list_controlnet_types,
    resolve_controlnet_path,
)
from utils.modeling.hf_loaders import caption_image_chain, score_hpsv2
from utils.modeling.hf_reward import HFRewardPanel, HFRewardWeights
from utils.modeling.hf_scaffold import (
    CONFIG_ONLY_IGNORE,
    FACE_RESTORE_MODELS,
    UPSCALE_MODELS,
    has_local_weights,
    registry_summary,
    resolve_entries,
    role_counts,
    scaffold_registry,
)
from utils.modeling.model_paths import pretrained_catalog

# ==================================================================================================
# test_hf_control: test_hf_control
# ==================================================================================================


def test_controlnet_registry_has_extended_types():
    types = list_controlnet_types()
    for name in ("mlsd", "softedge", "seg", "normal", "hed", "canny_sdxl"):
        assert name in types
    assert len(CONTROLNET_REGISTRY) >= 12


def test_resolve_controlnet_path_returns_string():
    p = resolve_controlnet_path("canny")
    assert isinstance(p, str)
    assert p


def test_pil_softedge_proxy(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "in.png"
    out = tmp_path / "soft.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(src)
    written = extract_pil_proxy(str(src), str(out), "softedge")
    assert written == str(out)
    assert out.is_file()


# ==================================================================================================
# test_hf_loaders: test_hf_loaders
# ==================================================================================================


def test_caption_chain_heuristic_fallback(tmp_path: Path):
    img = tmp_path / "ref.png"
    from PIL import Image

    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(img)
    cap, backend = caption_image_chain(str(img), backends=[])
    assert cap == ""
    assert backend == ""


def test_score_hpsv2_returns_none_without_model():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert score_hpsv2(rgb, "a cat", device="cpu") is None


# ==================================================================================================
# test_hf_reward: test_hf_reward
# ==================================================================================================


def test_hf_reward_panel_empty_backends():
    # Zero every backend weight (including ones added after this test was
    # written) so the panel really has no active backends.
    panel = HFRewardPanel(weights=HFRewardWeights(**{f.name: 0.0 for f in fields(HFRewardWeights)}))
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    bd = panel.score_breakdown(rgb, prompt="test")
    assert bd.composite == 0.5
    assert bd.parts == {}


def test_hf_reward_panel_score_without_models():
    panel = HFRewardPanel(device="cpu")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    s = panel.score(rgb, prompt="a cat")
    assert 0.0 <= s <= 1.0


# ==================================================================================================
# test_hf_scaffold: test_hf_scaffold
# ==================================================================================================


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


def test_registry_summary_and_roles():
    s = registry_summary()
    assert int(s["total_registry"]) == len(scaffold_registry()) >= 140
    rc = role_counts()
    for role in ("vlm", "reward", "control", "depth", "text_encoder"):
        assert rc.get(role, 0) >= 1


def test_post_process_model_lists():
    assert {"Real-ESRGAN", "SwinIR-classical-x2"} <= set(UPSCALE_MODELS)
    assert {"CodeFormer", "GFPGAN"} <= set(FACE_RESTORE_MODELS)
