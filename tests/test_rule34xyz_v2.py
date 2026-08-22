"""Unit tests for rule34.xyz v2 API adapter (offline)."""

from __future__ import annotations

from scripts.scrape.rule34xyz_v2 import (
    _md5_for_post,
    file_url_from_post,
    post_from_api,
    tags_from_post,
)


def test_md5_deterministic():
    assert _md5_for_post(12345) == _md5_for_post(12345)
    assert len(_md5_for_post(12345)) == 32


def test_file_url_cdn_and_direct():
    post = {"id": 4683574, "files": {"100": [1, 2], "10": [0, 2]}}
    url, ext = file_url_from_post(post)
    assert ext == "mp4"
    assert "rule34xyz.b-cdn.net" in url
    assert "/4683/4683574/4683574.mov.mp4" in url

    post2 = {"id": 100, "files": {"10": [0, 2]}}
    url2, ext2 = file_url_from_post(post2)
    assert ext2 == "jpg"
    assert url2.startswith("https://rule34.xyz/posts/0/100/")


def test_tags_from_post_types():
    post = {
        "tags": [
            {"value": "1girl", "type": 1},
            {"value": "pokemon", "type": 2},
            {"value": "misty", "type": 4},
            {"value": "artist_x", "type": 8},
        ]
    }
    all_tags, artists, chars, copies = tags_from_post(post)
    assert "1girl" in all_tags
    assert "artist_x" in artists
    assert "misty" in chars
    assert "pokemon" in copies


def test_post_from_api():
    post = {
        "id": 42,
        "width": 800,
        "height": 600,
        "files": {"10": [0, 2]},
        "tags": [{"value": "animated", "type": 1}],
    }
    p = post_from_api(post)
    assert p is not None
    assert p.site == "rule34xyz"
    assert p.md5 == _md5_for_post(42)
    assert p.file_url.endswith(".pic.jpg")
    assert "animated" in p.tags
