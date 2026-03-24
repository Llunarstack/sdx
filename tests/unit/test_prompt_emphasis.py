"""Tests for ( ) / [ ] prompt emphasis → token weight vectors (sample + train alignment)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_parse_prompt_emphasis() -> None:
    from utils.prompt.prompt_emphasis import parse_prompt_emphasis

    c, segs = parse_prompt_emphasis("(hi) there")
    assert c == "hi there"
    assert (0, 2, 1.2) in segs or segs[0] == (0, 2, 1.2)

    c2, _ = parse_prompt_emphasis("[soft] tail")
    assert c2 == "soft tail"


def test_token_weights_from_cleaned_segments_mock_tokenizer() -> None:
    from utils.prompt.prompt_emphasis import parse_prompt_emphasis, token_weights_from_cleaned_segments

    cleaned, segs = parse_prompt_emphasis("(hi) there")
    # 3 real tokens + padding to max_length=8
    om_row = [(0, 2), (2, 3), (3, 8)] + [(0, 0)] * 5

    class Tok:
        def __call__(self, batch, **kwargs):
            assert batch[0] == cleaned
            return {"offset_mapping": torch.tensor([om_row], dtype=torch.long)}

    w = token_weights_from_cleaned_segments(cleaned, segs, Tok(), max_length=8)
    assert w is not None
    assert w.shape == (8,)
    assert abs(float(w[0].item()) - 1.2) < 1e-5
    assert float(w[1].item()) == 1.0
    assert float(w[2].item()) == 1.0
    for i in range(3, 8):
        assert float(w[i].item()) == 1.0  # padding


def test_batch_encoder_token_weights_triple_appends_two_ones() -> None:
    from utils.prompt.prompt_emphasis import batch_encoder_token_weights

    om_row = [(0, 1)] + [(0, 0)] * 7

    class Tok:
        def __call__(self, batch, **kwargs):
            return {"offset_mapping": torch.tensor([om_row], dtype=torch.long)}

    bundle = SimpleNamespace(mode="triple", fusion=SimpleNamespace())
    caps = ["(a)"]
    cleaned, tw = batch_encoder_token_weights(
        caps, Tok(), max_length=8, device=torch.device("cpu"), dtype=torch.float32, text_bundle=bundle
    )
    assert cleaned == ["a"]
    assert tw is not None and tw.shape == (1, 10)
    assert tw[0, 0].item() == pytest.approx(1.2)
    assert float(tw[0, 8].item()) == 1.0 and float(tw[0, 9].item()) == 1.0
