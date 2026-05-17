from __future__ import annotations

from pathlib import Path

from PIL import Image
from utils.generation.image_dissection import parse_part_requests, visual_facts_from_requests


def test_parse_part_requests_simple() -> None:
    p = "use the hat from image 1 and take background from ref2"
    reqs = parse_part_requests(p)
    assert len(reqs) >= 2
    assert any(r.part.lower().strip() == "hat" and r.source_index == 0 for r in reqs)
    assert any(r.part.lower().strip() == "background" and r.source_index == 1 for r in reqs)


def test_visual_facts_mentions_parts() -> None:
    p = "use the sword from image 2"
    reqs = parse_part_requests(p)
    facts = visual_facts_from_requests(p, reqs, num_reference_images=2)
    joined = "\n".join(facts).lower()
    assert "sword" in joined
    assert "reference image 2" in joined


def test_dissect_background_full_mask(tmp_path: Path) -> None:
    # Heavy models are not required for background extraction.
    from utils.generation.image_dissection import dissect_images_to_parts

    img_p = tmp_path / "a.png"
    Image.new("RGB", (64, 32), (10, 20, 30)).save(img_p)

    reqs, parts, facts = dissect_images_to_parts(
        "take background from image 1",
        [str(img_p)],
        output_dir=tmp_path / "out",
        enable_heavy_models=True,  # still should work without model weights
    )
    assert reqs
    assert parts and parts[0].mask_path and parts[0].crop_path
    assert any("background" in f.lower() for f in facts)

