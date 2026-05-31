"""Tests for text encoder stack readiness helpers."""

from utils.modeling.text_encoder_stack import (
    PENTA_CATALOG,
    TRIPLE_CATALOG,
    catalog_for_mode,
    stack_status,
    stack_status_lines,
)


def test_catalog_for_mode():
    assert catalog_for_mode("penta") == PENTA_CATALOG
    assert catalog_for_mode("triple") == TRIPLE_CATALOG
    assert catalog_for_mode("t5") == ("T5-XXL",)


def test_stack_status_returns_slots():
    st = stack_status("penta")
    assert st.mode == "penta"
    assert len(st.slots) == 5
    assert {s.name for s in st.slots} == set(PENTA_CATALOG)


def test_stack_status_lines_markdown():
    lines = stack_status_lines("triple")
    assert any("triple" in ln for ln in lines)
    assert any("CLIP-ViT-L-14" in ln for ln in lines)
