from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


TORCH_AVAILABLE = _torch_available()

OPTIONAL_HEAVY_DEPS = (
    "torch",
    "torchvision",
    "xformers",
    "triton",
    "transformers",
    "diffusers",
    "accelerate",
    "timm",
    "cv2",
)

TORCH_DEPENDENT_TEST_BASENAMES = {
    "test_hybrid_dit_vit_generate.py",
    "test_naming_compat.py",
}


def pytest_ignore_collect(collection_path, config) -> bool:  # type: ignore[no-untyped-def]
    """Cheap pre-import skip for obviously torch-dependent test modules.

    Optimization only; the authoritative safety net is
    ``pytest_make_collect_report`` below, which catches any missing optional
    heavy dependency surfaced during collection (including deep transitive
    imports this textual scan can't see).
    """
    if TORCH_AVAILABLE:
        return False

    p = Path(str(collection_path))
    if p.suffix != ".py":
        return False

    if p.name in TORCH_DEPENDENT_TEST_BASENAMES:
        return True

    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    if "import torch" in txt or "torch." in txt:
        return True

    return False


def _missing_optional_dep(longrepr_text: str) -> str | None:
    """Return the optional dep name if a collection error is a missing-import for it."""
    if "No module named" not in longrepr_text:
        return None
    for dep in OPTIONAL_HEAVY_DEPS:
        if f"named '{dep}'" in longrepr_text or f"named '{dep}." in longrepr_text:
            return dep
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):  # type: ignore[no-untyped-def]
    """Convert collection failures caused by a missing optional heavy dependency
    into a skip instead of a hard error.
    """
    outcome = yield
    if TORCH_AVAILABLE:
        return

    report = outcome.get_result()
    if report.outcome != "failed":
        return

    dep = _missing_optional_dep(str(report.longrepr))
    if dep is None:
        return

    report.outcome = "skipped"
    path = getattr(collector, "path", None) or getattr(collector, "fspath", "")
    report.longrepr = (str(path), 0, f"Skipped: optional dependency '{dep}' not installed")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[no-untyped-def]
    """Skip (don't fail) tests that hit a missing optional heavy dependency at
    run time via an import deferred inside the test body.

    Mirrors ``pytest_make_collect_report`` for the case where the offending
    ``import torch`` only executes when the test runs. Only genuine
    ``ModuleNotFoundError`` for a known optional dep is converted -- real
    failures are left untouched.
    """
    outcome = yield
    if TORCH_AVAILABLE:
        return
    excinfo = outcome.excinfo
    if excinfo is None:
        return
    exc = excinfo[1]
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None) in OPTIONAL_HEAVY_DEPS:
        outcome.force_exception(pytest.skip.Exception(f"optional dependency '{exc.name}' not installed"))
