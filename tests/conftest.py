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

# Note: These are now also in ARCHIVED_MODULE_TEST_BASENAMES
TORCH_DEPENDENT_TEST_BASENAMES = {
    "test_hybrid_dit_vit_generate.py",
}

# Tests that depend on archived modules or unbuilt native extensions
ARCHIVED_MODULE_TEST_BASENAMES = {
    "test_agentic_stack.py",
    "test_ar_masks_extended.py",
    "test_book_helpers.py",
    "test_book_model_readiness.py",
    "test_book_training_helpers.py",
    "test_book_train_preset.py",
    "test_cfg_batched.py",
    "test_checkpoint_analysis.py",
    "test_ckpt_text_stack.py",
    "test_eval_report.py",
    "test_generation_pkg_exports.py",
    "test_hf_control.py",
    "test_hf_index.py",
    "test_hf_loaders.py",
    "test_hf_reward.py",
    "test_hf_scaffold.py",
    "test_hf_upscale.py",
    "test_hybrid_dit_vit_generate.py",
    "test_jsonl_caption_hygiene_native.py",
    "test_jsonutil.py",
    "test_manifest_gate_tool.py",
    "test_model_forward.py",
    "test_model_paths_gen_searcher.py",
    "test_multi_encoder_encode.py",
    "test_native_fast_paths.py",
    "test_plain_dict_snapshot.py",
    "test_prompt_ops_native.py",
    "test_prompt_training_pkg_lazy.py",
    "test_runtime_profiling.py",
    "test_simple_latent_generate.py",
    "test_style_native.py",
    "test_superior_extended.py",
    "test_superior_stack.py",
    "test_superior_wave10.py",
    "test_superior_wave11.py",
    "test_superior_wave12.py",
    "test_superior_wave3.py",
    "test_superior_wave4.py",
    "test_superior_wave5.py",
    "test_superior_wave6.py",
    "test_superior_wave7.py",
    "test_superior_wave8.py",
    "test_superior_wave9.py",
    "test_text_encoder_penta.py",
    "test_text_encoder_stack.py",
    "test_visual_brain.py",
    "test_visual_design.py",
    "test_visual_design_full.py",
}


def pytest_ignore_collect(collection_path, config) -> bool:  # type: ignore[no-untyped-def]
    """Pre-import skip for test modules that depend on archived modules or missing deps.

    Skips:
    1. Torch-dependent tests (when torch unavailable)
    2. Tests depending on archived modules (utils._archive/*, sdx_native)
    """
    p = Path(str(collection_path))
    if p.suffix != ".py":
        return False

    # Always skip tests depending on archived modules
    if p.name in ARCHIVED_MODULE_TEST_BASENAMES:
        return True

    # Skip torch-dependent tests when torch unavailable
    if not TORCH_AVAILABLE and p.name in TORCH_DEPENDENT_TEST_BASENAMES:
        return True

    if not TORCH_AVAILABLE:
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
