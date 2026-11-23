"""Tests for vit_quality: scorer, text encoder, naming compat.
Also holds what used to be: test_vit_scorer, test_vit_text_encoder, test_naming_compat.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from diffusion.schedules import get_timestep_loss_weight as get_timestep_loss_weight_canonical
from models.dit_text import DiT_P_2_Text
from models.dit_variants import SuperiorViT
from PIL import Image
from utils.modeling.architecture import normalize_num_ar_blocks as normalize_num_ar_blocks_canonical
from utils.quality import test_time_pick
from vit_quality.model import (
    HandcraftedTextFeaturizer,
    ViTScorer,
    build_text_featurizer,
    build_vit_model,
    cached_vit_scorer,
    featurizer_from_config,
    load_vit_quality_checkpoint,
    text_feature_vector,
)
from vit_quality.train import ViTManifestDataset

# ==================================================================================================
# test_vit_scorer: ViTScorer: one checkpoint-driven inference path for infer CLI, pick-best and ViT mining.
# ==================================================================================================


REPO = Path(__file__).resolve().parents[1]


def _tiny_ckpt(tmp_path: Path) -> Path:
    cfg = {"model_name": "resnet10t", "image_size": 64, "hidden_dim": 32, "use_ar_conditioning": True}
    model = build_vit_model("resnet10t", pretrained=False, hidden_dim=32, use_ar_conditioning=True)
    path = tmp_path / "vit.pt"
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)
    return path


def test_scorer_batches_captions_tta_and_ar(tmp_path: Path) -> None:
    scorer = ViTScorer(_tiny_ckpt(tmp_path), "cpu")
    x = torch.stack([scorer.image_tensor(Image.new("RGB", (80, 50), c)) for c in ("red", "blue")])
    assert x.shape == (2, 3, 64, 64)
    one = scorer(x, "a red square", 2)
    many = scorer(x, ["a red square", "a blue square"], tta=True)
    assert one["quality_logit"].shape == many["quality_logit"].shape == (2,)
    assert many["embedding"].shape == (2, 32)


def test_pick_and_infer_cli_share_the_scorer(tmp_path: Path) -> None:
    ckpt = _tiny_ckpt(tmp_path)
    cached_vit_scorer.cache_clear()
    ctx = test_time_pick.PickContext(prompt="cat", vit_ckpt_path=str(ckpt))
    imgs = [np.full((48, 96, 3), v, dtype=np.uint8) for v in (40, 200)]
    scores = test_time_pick._score_vit_quality(imgs, ctx)
    assert scores is not None and len(scores) == 2 and all(0.0 <= s <= 1.0 for s in scores)

    for i in range(3):
        Image.new("RGB", (64, 64), (i * 80, 0, 0)).save(tmp_path / f"{i}.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps({"image_path": f"{i}.png", "caption": "x"}) + "\n" for i in range(3)))
    out = tmp_path / "scored.jsonl"
    npz = tmp_path / "emb.npz"
    cmd = [sys.executable, "-m", "vit_quality.infer", "--ckpt", str(ckpt), "--manifest-jsonl", str(manifest)]
    cmd += ["--device", "cpu", "--out", str(out), "--out-npz", str(npz), "--rank", "--top-k", "2"]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["vit_final_score"] >= rows[1]["vit_final_score"]
    assert np.load(npz, allow_pickle=True)["embeddings"].shape == (3, 32)

    from utils.agentic.quality_loop import score_image_vit

    assert score_image_vit(tmp_path / "0.png", "x", vit_ckpt=str(ckpt), device="cpu") != (0.5, 0.5)


# ==================================================================================================
# test_vit_text_encoder: Tests for the ViT evaluator's semantic text featurizer (vit_quality/model.py)
# ==================================================================================================


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


# ==================================================================================================
# test_naming_compat
# ==================================================================================================


def test_diffusion_sampling_holy_grail_presets_available():
    from diffusion.sampling import list_holy_grail_presets  # noqa: PLC0415

    presets = list_holy_grail_presets()
    assert isinstance(presets, list)
    assert len(presets) > 0


def test_ar_block_conditioning_canonical_available():
    assert normalize_num_ar_blocks_canonical(2) == 2
    assert normalize_num_ar_blocks_canonical(7) == -1


def test_diffusion_timestep_loss_weight_canonical():
    assert callable(get_timestep_loss_weight_canonical)


def test_model_canonical_aliases_expose_symbols():
    assert callable(DiT_P_2_Text)
    assert callable(SuperiorViT)


def test_vit_quality_canonical_package_exports_checkpoint_loader():
    assert callable(load_vit_quality_checkpoint)


def test_vit_quality_lazy_attr_origin_covers_all():
    """Each ``vit_quality.__all__`` name must have ``_ATTR_ORIGIN``."""
    import vit_quality as vq

    for n in vq.__all__:
        assert n in vq._ATTR_ORIGIN, f"add _ATTR_ORIGIN[{n!r}] in vit_quality/__init__.py"


def test_config_defaults_prompt_style_mediums_surface():
    """Smoke: canonical config modules used by sample/prompt tools stay importable."""
    from config import presets, prompts, styles  # noqa: PLC0415

    assert isinstance(getattr(prompts, "WATERMARK_NEGATIVE_STRONG", None), str)
    assert callable(getattr(styles, "guidance_fragments", None))
    assert callable(getattr(styles, "style_guidance_fragments", None))
    assert callable(getattr(presets, "apply_preset_to_args", None))
