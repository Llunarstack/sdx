"""Booru scraper: safety gate, secrets parsing, and post normalization (no network)."""

from __future__ import annotations

from pathlib import Path

from scripts.scrape.booru_client import Post
from scripts.scrape.safety import blocked_query_tags, blocking_tags, is_allowed, normalize_tag
from scripts.scrape.scrape_cli import _parse_ratings
from scripts.scrape.secrets_config import parse_secrets_file
from scripts.scrape.sites import DanbooruAdapter, E621Adapter, Rule34xxxAdapter, build_adapter


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
    assert _parse_ratings("all") is None
    assert _parse_ratings("") is None
    # "safe"/"s" spans danbooru general+sensitive so it works across sites.
    assert _parse_ratings("s") == {"s", "g"}
    assert _parse_ratings("safe, questionable") == {"s", "g", "q"}
    assert _parse_ratings("g") == {"g"}
    assert _parse_ratings("e") == {"e"}


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
    from scripts.scrape.secrets_config import SiteCredentials

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
    from scripts.scrape.secrets_config import SiteCredentials

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
    from scripts.scrape.secrets_config import SiteCredentials

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
    from scripts.scrape.secrets_config import SiteCredentials

    try:
        build_adapter("pixiv", SiteCredentials(site="pixiv"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
