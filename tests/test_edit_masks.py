"""Tests for heuristic inpaint masks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from utils.generation.edit_masks import (
    HEURISTIC_EDIT_REGIONS,
    heuristic_inpaint_mask,
    normalize_heuristic_region,
    save_heuristic_mask,
)


def test_region_keys():
    assert "face" in HEURISTIC_EDIT_REGIONS
    assert "background" in HEURISTIC_EDIT_REGIONS


def test_normalize_unknown_defaults_to_subject():
    assert normalize_heuristic_region("not_a_real_region") == "subject"
    assert normalize_heuristic_region(" FACE ") == "face"


def test_full_all_white_stats():
    m = heuristic_inpaint_mask(64, 64, "full")
    assert m.size == (64, 64)
    mn, mx = m.getextrema()
    assert mx == 255


def test_background_masks_edges_brighter_than_center():
    m = heuristic_inpaint_mask(64, 64, "background").load()
    # corners should tendency to white (paint)
    corners = sum(m[x, y] for x, y in ((0, 0), (63, 0), (0, 63), (63, 63))) / 4
    cen = sum(m[32, y] for y in range(20, 44)) / 24
    assert corners > cen


def test_face_has_upper_center_activity():
    m = heuristic_inpaint_mask(96, 96, "face").load()
    hi = sum(m[48, y] for y in range(5, 30)) / 25
    lo = sum(m[48, y] for y in range(80, 95)) / 15
    assert hi > lo


def test_save_roundtrip(tmp_path: Path):
    p = tmp_path / "mask.png"
    save_heuristic_mask(p, 32, 32, "subject")
    assert p.is_file()
    Image.open(p).verify()
