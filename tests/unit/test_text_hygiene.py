"""sdx_native.text_hygiene — Unicode caption cleanup and fingerprints."""

from __future__ import annotations

import hashlib

from sdx_native.text_hygiene import (
    caption_fingerprint,
    normalize_caption_for_training,
    pos_neg_token_overlap,
    strip_zwsp,
)


def test_strip_zwsp_removes_zero_width():
    s = "hello\u200bworld"
    assert strip_zwsp(s) == "helloworld"


def test_normalize_caption_nfkc_and_segments():
    raw = "  foo  , \u200bbar , baz, ,"
    out = normalize_caption_for_training(raw)
    assert out == "foo, bar, baz"


def test_caption_fingerprint_sha256_when_no_xxhash():
    cap = "1girl, red dress"
    fp = caption_fingerprint(cap, algorithm="sha256")
    norm = normalize_caption_for_training(cap)
    assert fp == hashlib.sha256(norm.encode("utf-8")).hexdigest()


def test_pos_neg_token_overlap():
    inter, union, j = pos_neg_token_overlap("cat, dog, sky", "dog, watermark")
    assert inter == 1
    assert union == 4
    assert 0 < j < 1
