"""ViTScorer: one checkpoint-driven inference path for infer CLI, pick-best and ViT mining."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from utils.quality import test_time_pick
from vit_quality.checkpoint_utils import ViTScorer, cached_vit_scorer
from vit_quality.model import build_vit_model

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

    from utils.superior.auto_loop import score_image_vit

    assert score_image_vit(tmp_path / "0.png", "x", vit_ckpt=str(ckpt), device="cpu") != (0.5, 0.5)
