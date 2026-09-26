import numpy as np
from utils.quality.variation_guard import audit_variation, reseed_indices


def _rand_image(seed: int) -> np.ndarray:
    # Structurally distinct images (different composition + palette per seed);
    # iid noise would downsample to identical luma and correctly read as similar.
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 63.0
    angle = rng.uniform(0, np.pi)
    ramp = np.cos(angle) * xx + np.sin(angle) * yy
    base = rng.uniform(40, 200, size=3)
    for c in range(3):
        img[..., c] = base[c] + 80.0 * ramp
    x0, y0 = rng.integers(0, 40, size=2)
    img[y0 : y0 + 20, x0 : x0 + 20] = rng.uniform(0, 255, size=3)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_identical_batch_collapses():
    img = _rand_image(1)
    report = audit_variation([img.copy() for _ in range(4)])
    assert report.collapse_score > 0.9
    assert report.min_distance < 1e-6
    assert len(report.near_duplicate_pairs) == 6  # all pairs


def test_diverse_batch_is_healthy():
    report = audit_variation([_rand_image(s) for s in range(4)])
    assert report.collapse_score < 0.9
    assert not report.near_duplicate_pairs


def test_reseed_flags_duplicates_not_unique_images():
    base = _rand_image(1)
    batch = [base, base.copy(), _rand_image(2), _rand_image(3)]
    report = audit_variation(batch)
    flagged = reseed_indices(report, max_collapse=0.99)
    assert flagged in ([0], [1])  # one of the duplicate pair, not the unique images


def test_reseed_empty_for_diverse_batch():
    report = audit_variation([_rand_image(s) for s in range(4)])
    assert reseed_indices(report) == []


def test_single_image_batch_is_trivially_ok():
    report = audit_variation([_rand_image(0)])
    assert report.collapse_score == 0.0
    assert report.per_image_uniqueness == [0.0]
