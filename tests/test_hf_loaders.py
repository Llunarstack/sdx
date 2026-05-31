from pathlib import Path

import numpy as np
from utils.modeling.hf_loaders import caption_image_chain, score_hpsv2


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
