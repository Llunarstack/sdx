"""Tests for ``pipelines.book_comic.book_project`` layout."""

from __future__ import annotations

from pathlib import Path

from pipelines.book_comic.book_project import BookProject


def test_book_project_layout_and_flush(tmp_path: Path) -> None:
    proj = BookProject.open(tmp_path / "my_book", create=True)
    proj.title = "Test Vol 1"
    proj.ckpt = "results/run/best.pt"
    front = proj.cover_path("front")
    back = proj.cover_path("back")
    page0 = proj.page_path(0)
    front.write_bytes(b"png")
    back.write_bytes(b"png")
    page0.write_bytes(b"png")
    proj.add_entry({"kind": "front_cover", "path": proj.rel(front), "prompt": "front"})
    proj.add_entry({"kind": "back_cover", "path": proj.rel(back), "prompt": "back"})
    proj.add_entry({"kind": "page", "index": 0, "path": proj.rel(page0), "prompt": "p0"})
    proj.flush()
    proj.sync_legacy_front_cover(front)
    assert (tmp_path / "my_book" / "book.json").is_file()
    assert (tmp_path / "my_book" / "covers" / "front.png").is_file()
    assert (tmp_path / "my_book" / "covers" / "back.png").is_file()
    assert (tmp_path / "my_book" / "cover" / "cover.png").is_file()
    loaded = BookProject.open(tmp_path / "my_book", create=False)
    assert loaded.title == "Test Vol 1"
    assert len(loaded.entries) == 3


def test_book_project_open_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "book"
    BookProject.open(root, create=True).flush()
    again = BookProject.open(root, create=False)
    assert again.root == root.resolve()
