"""
Tests for the competitor-weakness eval harness (scripts/tools/eval/).

Verifies suite integrity, the reference blob counter's accuracy, the graded
scorers, and the runner's aggregation. CLIP-dependent paths are exercised only
when weights are available; everything else runs offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw
from scripts.tools.eval import scorers
from scripts.tools.eval.prompt_suite import CATEGORIES, SUITE, items_by_category, suite_as_manifest_rows
from scripts.tools.eval.run_eval import evaluate


def test_suite_integrity():
    ids = [s.id for s in SUITE]
    assert len(ids) == len(set(ids)), "duplicate suite ids"
    for s in SUITE:
        assert s.category in CATEGORIES, f"{s.id}: unknown category {s.category}"
        if s.category == "count":
            assert s.expected_count is not None, f"{s.id}: count item needs expected_count"
        if s.category == "text_render":
            assert s.expected_text, f"{s.id}: text item needs expected_text"
    # Manifest export round-trips the objective fields.
    rows = suite_as_manifest_rows()
    assert len(rows) == len(SUITE)
    assert any("expected_count" in r for r in rows)
    assert any("expected_text" in r for r in rows)


def _discs(path: Path, n: int, bg="white", fg="red") -> None:
    im = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(im)
    xs = [int(256 * (i + 1) / (n + 1)) for i in range(n)]
    for cx in xs:
        d.ellipse([cx - 14, 118, cx + 14, 146], fill=fg)
    im.save(path)


def test_blob_counter_accuracy():
    with tempfile.TemporaryDirectory() as td:
        for n in (1, 2, 3, 5):
            p = Path(td) / f"n{n}.png"
            _discs(p, n)
            assert scorers.count_objects(str(p)) == n, f"expected {n} blobs"
        # Dark-background polarity is handled too.
        p = Path(td) / "dark.png"
        _discs(p, 4, bg="black", fg="yellow")
        assert scorers.count_objects(str(p)) == 4


def test_count_score_decay():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "three.png"
        _discs(p, 3)
        s_exact, d = scorers.count_score(str(p), 3)
        assert s_exact == 1.0 and d["detected"] == 3
        s_off, d2 = scorers.count_score(str(p), 5)  # detected 3, expected 5 -> err 2
        assert d2["abs_error"] == 2
        assert 0.0 < s_off < s_exact


def test_scene_physics_scorers_optional():
    # Must not raise when utils.quality.scene_physics is absent.
    score, detail = scorers.occlusion_score("missing.png")
    assert score is None or isinstance(score, float)
    assert isinstance(detail, dict)
    score2, detail2 = scorers.reflection_score("missing.png")
    assert score2 is None or isinstance(score2, float)
    assert isinstance(detail2, dict)
    score3, detail3 = scorers.spatial_bind_score("missing.png", "a red cube to the left of a blue sphere")
    assert score3 is None or isinstance(score3, float)
    assert isinstance(detail3, dict)


def test_text_score_normalization():
    # LCS ratio helper is deterministic and offline.
    assert scorers._normalize_text("O.P-E N!") == "open"
    assert scorers._lcs_ratio("open", "the open sign") == 1.0
    assert scorers._lcs_ratio("open", "xxxx") == 0.0


def test_runner_aggregates():
    if not scorers.availability()["clip_adherence"]:
        import pytest

        pytest.skip("CLIP weights unavailable")
    with tempfile.TemporaryDirectory() as td:
        # Build images for one count item and one long-adherence item.
        count_item = items_by_category("count")[0]
        long_item = items_by_category("long_adherence")[0]
        cimg = Path(td) / "c.png"
        _discs(cimg, count_item.expected_count)
        limg = Path(td) / "l.png"
        Image.new("RGB", (256, 256), (30, 40, 90)).save(limg)
        report = evaluate(
            {count_item.id: str(cimg), long_item.id: str(limg)},
            clip_model="openai/clip-vit-base-patch32",
            device="cpu",
        )
        assert report["n_scored"] == 2
        assert report["overall_adherence"] is not None
        assert "count" in report["per_category"]
        # The count image matches expected exactly.
        assert report["per_category"]["count"]["count_score"] == 1.0


if __name__ == "__main__":
    test_suite_integrity()
    print("[ok] suite integrity")
    test_blob_counter_accuracy()
    print("[ok] blob counter accuracy")
    test_count_score_decay()
    print("[ok] count score decay")
    test_text_score_normalization()
    print("[ok] text normalization")
    if scorers.availability()["clip_adherence"]:
        test_runner_aggregates()
        print("[ok] runner aggregates")
    else:
        print("[skip] runner (no CLIP)")
    print("\nEval harness verified.")
