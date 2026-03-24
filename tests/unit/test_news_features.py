"""Landscape / IMPROVEMENTS helpers: RAG prompt, character lock, pick scores, resolution buckets."""

import numpy as np
from PIL import Image

from data.bucket_batch_sampler import ResolutionBucketBatchSampler
from data.t2i_dataset import Text2ImageDataset


def test_parse_resolution_buckets_train():
    from train import parse_resolution_buckets as prb

    assert prb(None) is None
    assert prb("") is None
    assert prb("256,512") == [(256, 256), (512, 512)]
    assert prb("512x768") == [(512, 768)]


def test_rag_merge():
    from utils.prompt.rag_prompt import merge_facts_into_prompt

    out = merge_facts_into_prompt("draw a car", ["Model X 2026", "red"])
    assert "Model X" in out
    assert "draw a car" in out


def test_character_lock():
    from utils.consistency.character_lock import extract_character_tag, merge_character_into_caption

    c = merge_character_into_caption("in a cafe", "h1", "short black hair", mode="prefix")
    assert "character_ref=h1" in c
    assert extract_character_tag(c) == "h1"


def test_score_exposure():
    from utils.quality.test_time_pick import score_exposure_balance

    flat = np.full((8, 8, 3), 128, dtype=np.uint8)
    hi = np.full((8, 8, 3), 255, dtype=np.uint8)
    assert score_exposure_balance(flat) > score_exposure_balance(hi)


def test_bucket_sampler_groups(tmp_path):
    """Indices in each batch share the same bucket id."""
    manifest = tmp_path / "m.jsonl"
    lines = []
    for i in range(12):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (128, 128), color=(i * 10, 0, 0)).save(p)
        lines.append(f'{{"image_path": "{p.name}", "caption": "a"}}')
    manifest.write_text("\n".join(lines), encoding="utf-8")
    ds = Text2ImageDataset(
        str(manifest),
        image_size=64,
        resolution_buckets=[(64, 64), (32, 32)],
        bucket_seed=0,
        bucket_fixed_assign=True,
    )
    ds.set_epoch(0)
    bs = ResolutionBucketBatchSampler(ds, batch_size=2, drop_last=True)
    batches = list(iter(bs))
    assert len(batches) >= 1
    for b in batches:
        bids = {ds._bucket_assign[i] for i in b}
        assert len(bids) == 1


def test_quality_motion_glare():
    from utils.quality import add_lens_glare, add_motion_blur

    x = np.zeros((16, 16, 3), dtype=np.uint8)
    y = add_motion_blur(x, amount=0.2, angle_deg=0.0, seed=1)
    assert y.shape == x.shape
    z = add_lens_glare(x, strength=0.15, seed=2)
    assert z.mean() >= x.mean()
