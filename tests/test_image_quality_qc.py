from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def _checkerboard(size: int = 64, squares: int = 8) -> Image.Image:
    img = Image.new("L", (size, size), 0)
    px = img.load()
    step = size // squares
    for y in range(size):
        for x in range(size):
            v = 255 if ((x // step) + (y // step)) % 2 == 0 else 0
            px[x, y] = v
    return img.convert("RGB")


@pytest.fixture
def sharp_blur_imgs(tmp_path: Path) -> tuple[Path, Path]:
    sharp = tmp_path / "sharp.png"
    blur = tmp_path / "blur.png"
    img = _checkerboard(64, squares=8)
    img.save(sharp)
    img.filter(ImageFilter.GaussianBlur(radius=2.5)).save(blur)
    return sharp, blur


def test_image_quality_qc_thresholding(tmp_path: Path, sharp_blur_imgs: tuple[Path, Path]) -> None:
    sharp, blur = sharp_blur_imgs
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"image_path": str(sharp), "caption": "x"},
        {"image_path": str(blur), "caption": "x"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    script = ROOT / "scripts" / "tools" / "image_quality_qc.py"
    cmd_ok = [sys.executable, str(script), str(manifest), "--min-sharpness", "0.0", "--min-contrast", "0.0"]
    r = subprocess.run(cmd_ok, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr

    # Use a threshold that should reject the blurred image.
    # This is heuristic; the sharp checkerboard is expected to have much higher laplacian variance.
    cmd_fail = [sys.executable, str(script), str(manifest), "--min-sharpness", "2000", "--min-contrast", "0.0"]
    r2 = subprocess.run(cmd_fail, capture_output=True, text=True, timeout=30)
    assert r2.returncode == 1, r2.stderr
    assert "imageqc:" in r2.stdout

