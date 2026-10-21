"""Tests for SauceNAO + TinEye reverse search helpers."""

from __future__ import annotations

from utils.caption.reverse_search import ReverseHit, hit_meets_threshold, parse_source_url


def test_parse_danbooru_url():
    site, pid = parse_source_url("https://danbooru.donmai.us/posts/1234567")
    assert site == "danbooru"
    assert pid == "1234567"


def test_parse_e621_url():
    site, pid = parse_source_url("https://e621.net/posts/987654")
    assert site == "e621"
    assert pid == "987654"


def test_parse_pixiv_artworks_url():
    site, pid = parse_source_url("https://www.pixiv.net/en/artworks/555")
    assert site == "pixiv"
    assert pid == "555"


def test_tineye_booru_hit_lower_threshold():
    hit = ReverseHit(similarity=55.0, site="danbooru", site_id="1", engine="tineye")
    assert hit_meets_threshold(hit, 75.0) is True


def test_saucenao_html_parser():
    html = """
    <div class="resultadosub"><div class="resulttable"><div>87.3%</div>
    <a href="https://danbooru.donmai.us/posts/12345">danbooru</a>
    Characters: hatsune_miku</div></div>
    """
    from utils.caption.reverse_search import _hits_from_saucenao_html

    hits = _hits_from_saucenao_html(html)
    assert hits
    assert hits[0].site == "danbooru"
    assert hits[0].site_id == "12345"
    assert hits[0].similarity == 87.3


def test_saucenao_hit_needs_full_threshold():
    hit = ReverseHit(similarity=55.0, site="danbooru", site_id="1", engine="saucenao")
    assert hit_meets_threshold(hit, 75.0) is False


def test_fetch_booru_post_parses_danbooru_and_e621(monkeypatch):
    import requests
    from utils.caption.reverse_search import fetch_booru_post

    payloads = {
        "danbooru": {"tag_string_character": "hatsune_miku", "tag_string_artist": "kei", "tag_string": "a b"},
        "e621": {"post": {"tags": {"character": ["renamon"], "general": ["fur", " "], "meta": "x"}}},
    }

    class _Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_get(url, **kwargs):
        return _Resp(payloads["danbooru" if "danbooru" in url else "e621"])

    monkeypatch.setattr(requests, "get", fake_get)
    db = fetch_booru_post("danbooru", "12")
    assert db.character_tags == ["hatsune_miku"] and db.artist_tags == ["kei"] and db.tag_string == "a b"
    e6 = fetch_booru_post("e621", "34")
    assert e6.character_tags == ["renamon"] and e6.general_tags == ["fur"] and e6.tag_string == "renamon fur"
    assert fetch_booru_post("pixiv", "1") is None
    assert fetch_booru_post("danbooru", "abc") is None
