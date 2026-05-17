"""Lazy package layout: nested ``utils.*`` packages avoid eager submodule imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# OpenMP duplicate-runtime abort (MKL + other libs) in one-shot ``python -c`` smokes.
_SUBPROC_ENV = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}


def test_utils_prompt_import_does_not_load_sibling_modules() -> None:
    code = r"""
import sys
import utils.prompt as p

loaded = sorted(k for k in sys.modules if k.startswith("utils.prompt.") and k != "utils.prompt")
assert loaded == [], f"unexpected eager loads: {loaded}"
# Smoke: lazy attribute resolves submodule
assert p.neg_filter.filter_negative_by_positive("a", "a, b") == "b"
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_utils_training_import_does_not_load_sibling_modules() -> None:
    code = r"""
import sys
import utils.training as t

loaded = sorted(k for k in sys.modules if k.startswith("utils.training.") and k != "utils.training")
assert loaded == [], f"unexpected eager loads: {loaded}"
assert t.ar_curriculum.normalize_ar_blocks(1) == 2
assert "utils.training.ar_curriculum" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_utils_modeling_import_does_not_load_sibling_modules() -> None:
    code = r"""
import sys
import utils.modeling as m

loaded = sorted(k for k in sys.modules if k.startswith("utils.modeling.") and k != "utils.modeling")
assert loaded == [], f"unexpected eager loads: {loaded}"
_root = m.model_paths.repo_root()
assert isinstance(_root.as_posix(), str)
assert "utils.modeling.model_paths" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_utils_flat_packages_lazy_checkpoint_analysis_consistency_arch_quality() -> None:
    code = r"""
import importlib
import sys

def assert_no_submodules_loaded(pkg: str) -> None:
    importlib.import_module(pkg)
    base = pkg + "."
    bad = [k for k in sys.modules if k.startswith(base) and k != pkg]
    assert not bad, bad

assert_no_submodules_loaded("utils.checkpoint")
import utils.checkpoint as ck

ck.checkpoint_loading
assert "utils.checkpoint.checkpoint_loading" in sys.modules

assert_no_submodules_loaded("utils.analysis")
import utils.analysis as an

an.data_analysis
assert "utils.analysis.data_analysis" in sys.modules

assert_no_submodules_loaded("utils.consistency")
import utils.consistency as co

co.consistency_system
assert "utils.consistency.consistency_system" in sys.modules

assert_no_submodules_loaded("utils.architecture")
import utils.architecture as ar

ar.architecture_map
assert "utils.architecture.architecture_map" in sys.modules

assert_no_submodules_loaded("utils.quality")
import utils.quality as q

assert callable(q.naturalize)
assert "utils.quality.quality" in sys.modules

from utils import sharpen

import numpy as np

x = np.zeros((1, 1, 3), dtype=np.uint8)
sharpen(x, amount=0.0)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_utils_visual_design_import_does_not_load_sibling_modules() -> None:
    code = r"""
import sys
import utils.visual_design as vd

loaded = sorted(k for k in sys.modules if k.startswith("utils.visual_design.") and k != "utils.visual_design")
assert loaded == [], f"unexpected eager loads: {loaded}"
_ = vd.design_pack_ids()
assert "utils.visual_design.compose" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_vit_quality_import_does_not_load_sibling_modules() -> None:
    code = r"""
import sys
import vit_quality as vq

loaded = sorted(k for k in sys.modules if k.startswith("vit_quality.") and k != "vit_quality")
assert loaded == [], f"unexpected eager loads: {loaded}"
_ = vq.ViTConfig
assert "vit_quality.config" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )


def test_utils_native_does_not_import_sdx_native_until_first_attr() -> None:
    code = r"""
import sys
import utils.native as n

assert "sdx_native" not in sys.modules
_ = n.normalize_caption_csv
assert "sdx_native" in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        check=True,
        env=_SUBPROC_ENV,
    )
