"""Tests for scripts.scrape: safety filters, frame split, media validation, site adapters.
Also holds what used to be: test_scrape_safety, test_frame_split, test_media_validate, test_post_cap, test_rule34xyz_v2.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from scripts.scrape import (
    DanbooruAdapter,
    E621Adapter,
    Post,
    Rule34xxxAdapter,
    _looks_like_video_file,
    _md5_for_post,
    blocked_query_tags,
    blocking_tags,
    build_adapter,
    extract_training_frames,
    file_url_from_post,
    is_allowed,
    is_blocked_download_ext,
    is_splittable_ext,
    needs_frame_split,
    normalize_tag,
    parse_ratings,
    parse_secrets_file,
    post_cap_reached,
    post_from_api,
    posts_remaining,
    sniff_media_kind,
    tags_from_post,
    validate_trainable_image,
)

# ==================================================================================================
# test_scrape_safety: Booru scraper: safety gate, secrets parsing, and post normalization (no network).
# ==================================================================================================


def test_blocklist_blocks_known_bad_tags():
    for bad in ("loli", "shota", "cub", "toddlercon", "young", "lolicon"):
        assert not is_allowed(["scenery", bad]), bad
        assert bad in blocking_tags(["scenery", bad])


def test_blocklist_allows_clean_tags():
    assert is_allowed(["landscape", "forest", "1girl", "solo"])
    assert blocking_tags(["cityscape", "detailed_background"]) == []


def test_blocklist_no_substring_false_positives():
    # "cub" must not blanket-block "cuba"/"incubator"/"cube".
    assert is_allowed(["cuba", "incubator", "cube_root"])


def test_normalize_tag_spaces_and_case():
    assert normalize_tag(" Loli Dominant ") == "loli_dominant"
    assert not is_allowed(["Loli"])  # case-insensitive


def test_blocked_query_tags_are_negated():
    q = blocked_query_tags()
    assert all(t.startswith("-") for t in q)
    assert "-loli" in q and "-shota" in q


def test_parse_ratings():
    assert parse_ratings("all") is None
    assert parse_ratings("") is None
    # "safe"/"s" spans danbooru general+sensitive so it works across sites.
    assert parse_ratings("s") == {"s", "g"}
    assert parse_ratings("safe, questionable") == {"s", "g", "q"}
    assert parse_ratings("g") == {"g"}
    assert parse_ratings("e") == {"e"}


def test_secrets_parsing(tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text(
        "danbooru\n\nuser: Alice\nip: DANKEY123\n\n"
        "e621:\n\nuser: Bob\np: pw\napi: E6KEY456\n\n"
        "rule34xxx\n\nuser: Carl\napi?: &api_key=abc123def&user_id=42\n",
        encoding="utf-8",
    )
    creds = parse_secrets_file(secret)
    assert creds["danbooru"].username == "Alice"
    assert creds["danbooru"].api_key == "DANKEY123"  # mislabeled 'ip' recovered
    assert creds["e621"].username == "Bob"
    assert creds["e621"].api_key == "E6KEY456"
    assert creds["rule34xxx"].api_key == "abc123def"
    assert creds["rule34xxx"].user_id == "42"


def test_danbooru_parse_and_auth():
    from scripts.scrape import SiteCredentials

    a = DanbooruAdapter(SiteCredentials(site="danbooru", username="u", api_key="k"))
    params = a.build_params("scenery", 1)
    assert params["login"] == "u" and params["api_key"] == "k" and params["tags"] == "scenery"
    posts = list(
        a.parse_posts(
            [
                {
                    "id": 5,
                    "md5": "abc",
                    "file_url": "https://x/y.png",
                    "file_ext": "png",
                    "tag_string": "landscape sky",
                    "rating": "s",
                    "image_width": 512,
                    "image_height": 512,
                }
            ]
        )
    )
    assert len(posts) == 1
    p = posts[0]
    assert p.md5 == "abc" and p.ext == "png" and p.tags == ["landscape", "sky"]
    assert p.caption == "landscape, sky"


def test_e621_basic_auth_and_nested_tags():
    from scripts.scrape import SiteCredentials

    a = E621Adapter(SiteCredentials(site="e621", username="u", api_key="k"))
    assert a.auth == ("u", "k")
    data = {
        "posts": [
            {
                "id": 9,
                "file": {"url": "https://e/z.jpg", "ext": "jpg", "md5": "m", "width": 100, "height": 80},
                "tags": {"general": ["forest", "tree"], "species": ["wolf"]},
                "rating": "s",
            }
        ]
    }
    posts = list(a.parse_posts(data))
    assert posts[0].tags == ["forest", "tree", "wolf"]
    assert posts[0].md5 == "m" and posts[0].width == 100


def test_rule34_url_reconstruction():
    from scripts.scrape import SiteCredentials

    a = Rule34xxxAdapter(SiteCredentials(site="rule34xxx", api_key="k", user_id="1"))
    assert a.first_page == 0
    posts = list(
        a.parse_posts([{"id": 1, "hash": "h", "directory": "12", "image": "pic.jpg", "tags": "a b", "rating": "e"}])
    )
    assert posts[0].file_url.endswith("/images/12/pic.jpg")
    assert posts[0].ext == "jpg"


def test_post_caption_underscores_to_spaces():
    p = Post(site="danbooru", id="1", md5="m", file_url="u", ext="png", tags=["long_hair", "blue_sky"])
    assert p.caption == "long hair, blue sky"


def test_build_adapter_rejects_unknown_site():
    from scripts.scrape import SiteCredentials

    try:
        build_adapter("pixiv", SiteCredentials(site="pixiv"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ==================================================================================================
# test_frame_split: Smoke tests for GIF/video frame splitting.
# ==================================================================================================


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


# ==================================================================================================
# test_media_validate: Tests for downloaded media validation.
# ==================================================================================================


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


# ==================================================================================================
# test_post_cap
# ==================================================================================================


def test_unlimited_cap():
    assert not post_cap_reached(0, 0)
    assert not post_cap_reached(999999, 0)


def test_finite_cap():
    assert not post_cap_reached(4, 5)
    assert post_cap_reached(5, 5)
    assert posts_remaining(10, 3) == 7
    assert posts_remaining(0, 3) == 0


# ==================================================================================================
# test_rule34xyz_v2: Unit tests for rule34.xyz v2 API adapter (offline).
# ==================================================================================================


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
