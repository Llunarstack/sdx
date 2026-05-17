from __future__ import annotations

import importlib.util
from pathlib import Path


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


TORCH_AVAILABLE = _torch_available()

TORCH_DEPENDENT_TEST_BASENAMES = {
    # These tests don't explicitly contain "import torch", but they import
    # torch-dependent modules during test module import.
    "test_hybrid_dit_vit_generate.py",
    "test_naming_compat.py",
}


def pytest_ignore_collect(collection_path, config) -> bool:  # type: ignore[no-untyped-def]
    """
    If torch isn't installed, skip torch-dependent tests early.

    This repo contains a mix of pure-python prompt/data utilities and
    torch-based numerical checks. Without torch, importing torch-based test
    modules fails during collection.
    """
    if TORCH_AVAILABLE:
        return False

    p = Path(str(collection_path))
    if p.suffix != ".py":
        return False

    if p.name in TORCH_DEPENDENT_TEST_BASENAMES:
        return True

    # Fast allow: if the test file doesn't mention torch, let it try.
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    if "import torch" in txt or "torch." in txt:
        return True

    return False

