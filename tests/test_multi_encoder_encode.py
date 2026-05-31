"""Tests for multi-encoder encode kwargs helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.modeling.multi_encoder_encode import (
    clip_and_long_captions_for_layout,
    encode_kwargs_for_captions,
    prepare_multi_encoder_kwargs,
)


@dataclass
class _FakeFusion:
    extra_token_count: int = 2


@dataclass
class _FakeBundle:
    mode: str
    fusion: Optional[_FakeFusion] = None


def test_encode_kwargs_t5_only():
    assert encode_kwargs_for_captions(["hello"], None) == {}
    assert encode_kwargs_for_captions(["hello"], _FakeBundle(mode="t5")) == {}


def test_encode_kwargs_triple():
    b = _FakeBundle(mode="triple", fusion=_FakeFusion())
    kw = encode_kwargs_for_captions(["a cat"], b)
    assert kw == {"clip_captions": ["a cat"]}
    assert "long_clip_captions" not in kw


def test_encode_kwargs_penta():
    b = _FakeBundle(mode="penta", fusion=_FakeFusion(extra_token_count=4))
    kw = encode_kwargs_for_captions(["long prompt text"], b)
    assert kw["clip_captions"] == ["long prompt text"]
    assert kw["long_clip_captions"] == ["long prompt text"]


def test_layout_aware_clip_caption():
    layout = {
        "preset_order": "subject_first",
        "subjects": ["red sports car"],
        "environment": "mountain road at sunset",
    }
    flat = "red sports car, mountain road at sunset, cinematic"
    clip_s, long_s = clip_and_long_captions_for_layout(flat, layout)
    assert "SUBJECT" in clip_s or "red sports car" in clip_s
    assert long_s == flat

    b = _FakeBundle(mode="penta", fusion=_FakeFusion(extra_token_count=4))
    kw = prepare_multi_encoder_kwargs(
        [flat],
        b,
        layout_specs=[layout],
        flat_full_prompts=[flat],
    )
    assert kw["clip_captions"][0] != flat or "SUBJECT" in kw["clip_captions"][0]
    assert kw["long_clip_captions"][0] == flat


def test_layout_specs_none_falls_back():
    b = _FakeBundle(mode="triple", fusion=_FakeFusion())
    kw = prepare_multi_encoder_kwargs(["plain"], b, layout_specs=[None])
    assert kw["clip_captions"] == ["plain"]
