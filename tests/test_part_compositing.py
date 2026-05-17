from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from utils.generation.image_dissection import DissectedPart, PartRequest
from utils.generation.part_compositing import build_init_and_inpaint_mask


def test_build_init_and_mask_preserves_foreground(tmp_path: Path) -> None:
    # Base background: solid gray
    bg = tmp_path / "bg.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(bg)

    # Foreground crop: red square everywhere, mask only in a small region.
    fg = tmp_path / "fg.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(fg)

    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 255
    mask_p = tmp_path / "m.png"
    Image.fromarray(m, mode="L").save(mask_p)

    part = DissectedPart(
        request=PartRequest(part="hat", source_index=0, role="foreground"),
        mask_path=str(mask_p),
        crop_path=str(fg),
        confidence=1.0,
    )

    spec = build_init_and_inpaint_mask(
        reference_images=[str(bg)],
        parts=[part],
        output_dir=tmp_path / "out",
        target_size=(32, 32),
        lock_background_if_requested=False,
    )

    init = Image.open(spec.init_image_path).convert("RGB")
    mask = Image.open(spec.mask_path).convert("L")
    init_np = np.array(init)
    mask_np = np.array(mask)

    # Center should be red and preserved (mask black).
    assert (init_np[16, 16] == np.array([255, 0, 0])).all()
    assert mask_np[16, 16] == 0
    # Corner should remain background and be inpaintable (mask white).
    assert (init_np[0, 0] == np.array([128, 128, 128])).all()
    assert mask_np[0, 0] == 255
