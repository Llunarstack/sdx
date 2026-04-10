"""Tests for ``pipelines.book_comic.book_manifest_utils``."""

from __future__ import annotations

from pipelines.book_comic.book_manifest_utils import (
    manifest_summary_lines,
    validate_book_manifest,
)


def test_validate_book_manifest_ok_minimal() -> None:
    m = {
        "ckpt": "models/x.pt",
        "entries": [{"kind": "page", "index": 0, "path": "pages/p000.png", "prompt": "a", "seed": 1}],
    }
    errs, warns = validate_book_manifest(m)
    assert not errs


def test_validate_book_manifest_missing_ckpt() -> None:
    errs, _ = validate_book_manifest({"entries": []})
    assert any("ckpt" in e.lower() for e in errs)


def test_validate_vit_pick_without_ckpt_warns() -> None:
    m = {
        "ckpt": "z.pt",
        "pick_best": "combo_vit_hq",
        "entries": [],
    }
    _, warns = validate_book_manifest(m)
    assert any("pick_vit" in w.lower() or "vit" in w.lower() for w in warns)


def test_manifest_summary_includes_pick() -> None:
    lines = manifest_summary_lines(
        {"pick_best": "combo", "sample_candidates": 2, "entries": [], "ckpt": "a.pt"}
    )
    assert any("pick_best=combo" in x for x in lines)
    assert any("sample_candidates=2" in x for x in lines)
