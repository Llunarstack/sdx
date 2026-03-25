"""Preference image dataset (paths + JSONL)."""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from utils.training.preference_image_dataset import PreferenceImageDataset


def test_preference_image_dataset_loads(tmp_path):
    img_dir = tmp_path / "im"
    img_dir.mkdir()
    w = img_dir / "w.png"
    l = img_dir / "l.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(w)
    Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8) * 200).save(l)
    j = tmp_path / "p.jsonl"
    j.write_text(
        json.dumps({"win_image_path": "w.png", "lose_image_path": "l.png", "caption": "test"}) + "\n",
        encoding="utf-8",
    )
    ds = PreferenceImageDataset(j, image_size=8, image_root=str(img_dir))
    row = ds[0]
    assert row["win"].shape == (3, 8, 8)
    assert row["lose"].shape == (3, 8, 8)
    assert row["prompt"] == "test"
