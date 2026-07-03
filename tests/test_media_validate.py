"""Tests for downloaded media validation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.scrape.media_validate import (
    is_blocked_download_ext,
    sniff_media_kind,
    validate_trainable_image,
)


def test_blocked_extensions():
    assert is_blocked_download_ext("zip")
    assert is_blocked_download_ext(".swf")
    assert not is_blocked_download_ext("jpg")


def test_sniff_html_and_zip(tmp_path: Path):
    html = tmp_path / "bad.jpg"
    html.write_text("<html>403</html>", encoding="utf-8")
    assert sniff_media_kind(html) == "html"

    z = tmp_path / "bad.zip"
    z.write_bytes(b"PK\x03\x04" + b"\x00" * 20)
    assert sniff_media_kind(z) == "zip"


def test_validate_png(tmp_path: Path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(p)
    assert validate_trainable_image(p)
