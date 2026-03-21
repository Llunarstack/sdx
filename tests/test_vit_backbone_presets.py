"""Tests for ViT.backbone_presets."""

import pytest

timm = pytest.importorskip("timm")

from ViT.backbone_presets import TIMM_BACKBONE_PRESETS, describe_presets_for_help  # noqa: E402


def test_describe_presets_non_empty():
    s = describe_presets_for_help()
    assert "vit_base" in s
    assert "timm" in s.lower()


@pytest.mark.parametrize("name,_why", TIMM_BACKBONE_PRESETS)
def test_timm_models_exist(name, _why):
    assert timm.is_model(name), f"missing timm model: {name}"
