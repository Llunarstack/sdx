"""Smoke tests for GIF/video frame splitting."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from scripts.scrape.frame_split import (
    _looks_like_video_file,
    extract_training_frames,
    is_splittable_ext,
    needs_frame_split,
)


def _write_animated_gif(path: Path, n: int = 4) -> None:
    frames = [Image.new("RGB", (32, 32), (40 * i, 20, 10)) for i in range(n)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )


def test_splittable_extensions():
    assert is_splittable_ext("gif")
    assert is_splittable_ext("mp4")
    assert is_splittable_ext("webm")
    assert not is_splittable_ext("jpg")


def test_gif_frame_split(tmp_path: Path):
    src = tmp_path / "anim.gif"
    _write_animated_gif(src)
    assert needs_frame_split(src, "gif")
    out_dir = tmp_path / "images"
    frames = extract_training_frames(src, out_dir, "deadbeef", "gif", max_frames=10)
    assert len(frames) >= 2
    for fr in frames:
        assert fr.abs_path.is_file()
        assert "deadbeef_f" in fr.rel_path
        assert "_f" in fr.rel_path


def test_rejects_html_disguised_as_video(tmp_path: Path):
    bad = tmp_path / "fake.mp4"
    bad.write_text("<html><body>403 Forbidden</body></html>", encoding="utf-8")
    assert not _looks_like_video_file(bad)
    frames = extract_training_frames(bad, tmp_path / "images", "cafebabe", "mp4")
    assert frames == []
