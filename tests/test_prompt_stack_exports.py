"""Regression: public prompt stack and config shims."""

from __future__ import annotations


def test_stack_exports_append_csv_and_token_set():
    from utils.prompt.stack import append_csv, token_set

    assert "b" in token_set("A, b")
    assert append_csv("a", "b") == "a, b"


def test_config_prompt_domains_shims():
    from config.defaults.prompt_domains import DEFAULT_NEGATIVE_PROMPT as canon
    from config.prompt_domains import DEFAULT_NEGATIVE_PROMPT as shim
    from config.reference.prompt_domains import DEFAULT_NEGATIVE_PROMPT as legacy

    assert shim == canon == legacy
