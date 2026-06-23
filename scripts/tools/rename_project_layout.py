"""One-shot project layout rename: innovations package, diffusion/sampling, agentic modules."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Top-level package rename ---
PACKAGE_RENAME = {"advanced_innovations": "innovations"}

# --- diffusion/sampling_extras → diffusion/sampling ---
SAMPLING_RENAME = "sampling"

AGENTIC_FILE_RENAMES = {
    "adaptive_learning_system.py": "adaptive_learning.py",
    "adversarial_robustness.py": "adversarial.py",
    "ensemble_validator.py": "ensemble.py",
    "evolving_quality_framework.py": "quality_framework.py",
    "explainable_quality_scoring.py": "explainable_scoring.py",
    "flow_matching_consistency.py": "flow_consistency.py",
    "generation_artifact_detector.py": "artifact_detector.py",
    "iterative_refinement_loop.py": "refinement_loop.py",
    "memory_preference_system.py": "memory_prefs.py",
    "perceptual_metrics_system.py": "perceptual_metrics.py",
    "prompt_adherence_system.py": "prompt_adherence.py",
    "prompt_optimization_agent.py": "prompt_optimizer.py",
    "quality_control_agent.py": "quality_control.py",
    "realtime_quality_monitor.py": "quality_monitor.py",
    "rlhf_agent.py": "rlhf.py",
    "semantic_composition_reasoner.py": "composition_reasoner.py",
    "semantic_drift_detector.py": "drift_detector.py",
    "vision_reward_system.py": "vision_reward.py",
    "visual_reasoning_agent.py": "visual_reasoning.py",
}

INNOVATIONS_FILE_RENAMES = {
    "integration.py": "pipeline.py",
}

TEST_RENAMES = {
    "test_advanced_innovations.py": "test_innovations.py",
    "test_sampling_extras.py": "test_sampling.py",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}

# Text replacements (longer patterns first)
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    ("advanced_innovations.integration", "innovations.pipeline"),
    ("advanced_innovations", "innovations"),
    ("diffusion.sampling_extras", "diffusion.sampling"),
    ("diffusion/sampling_extras", "diffusion/sampling"),
    ("tests/test_advanced_innovations.py", "tests/test_innovations.py"),
    ("tests/test_sampling_extras.py", "tests/test_sampling.py"),
    ("test_advanced_innovations", "test_innovations"),
    ("test_sampling_extras", "test_sampling"),
    # agentic module paths in imports / docs
    (".adaptive_learning_system", ".adaptive_learning"),
    (".adversarial_robustness", ".adversarial"),
    (".ensemble_validator", ".ensemble"),
    (".evolving_quality_framework", ".quality_framework"),
    (".explainable_quality_scoring", ".explainable_scoring"),
    (".flow_matching_consistency", ".flow_consistency"),
    (".generation_artifact_detector", ".artifact_detector"),
    (".iterative_refinement_loop", ".refinement_loop"),
    (".memory_preference_system", ".memory_prefs"),
    (".perceptual_metrics_system", ".perceptual_metrics"),
    (".prompt_adherence_system", ".prompt_adherence"),
    (".prompt_optimization_agent", ".prompt_optimizer"),
    (".quality_control_agent", ".quality_control"),
    (".realtime_quality_monitor", ".quality_monitor"),
    (".rlhf_agent", ".rlhf"),
    (".semantic_composition_reasoner", ".composition_reasoner"),
    (".semantic_drift_detector", ".drift_detector"),
    (".vision_reward_system", ".vision_reward"),
    (".visual_reasoning_agent", ".visual_reasoning"),
    ("adaptive_learning_system.py", "adaptive_learning.py"),
    ("adversarial_robustness.py", "adversarial.py"),
    ("ensemble_validator.py", "ensemble.py"),
    ("evolving_quality_framework.py", "quality_framework.py"),
    ("explainable_quality_scoring.py", "explainable_scoring.py"),
    ("flow_matching_consistency.py", "flow_consistency.py"),
    ("generation_artifact_detector.py", "artifact_detector.py"),
    ("iterative_refinement_loop.py", "refinement_loop.py"),
    ("memory_preference_system.py", "memory_prefs.py"),
    ("perceptual_metrics_system.py", "perceptual_metrics.py"),
    ("prompt_adherence_system.py", "prompt_adherence.py"),
    ("prompt_optimization_agent.py", "prompt_optimizer.py"),
    ("quality_control_agent.py", "quality_control.py"),
    ("realtime_quality_monitor.py", "quality_monitor.py"),
    ("rlhf_agent.py", "rlhf.py"),
    ("semantic_composition_reasoner.py", "composition_reasoner.py"),
    ("semantic_drift_detector.py", "drift_detector.py"),
    ("vision_reward_system.py", "vision_reward.py"),
    ("visual_reasoning_agent.py", "visual_reasoning.py"),
]


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def rename_top_level_package() -> None:
    for old, new in PACKAGE_RENAME.items():
        src = ROOT / old
        dst = ROOT / new
        if src.is_dir() and not dst.exists():
            src.rename(dst)
            print(f"package: {old} -> {new}")


def rename_innovations_files(pkg: Path) -> None:
    for old, new in INNOVATIONS_FILE_RENAMES.items():
        src = pkg / old
        dst = pkg / new
        if src.is_file() and not dst.exists():
            src.rename(dst)
            print(f"  innovations/{old} -> {new}")


def rename_agentic_files(agentic: Path) -> None:
    for old, new in AGENTIC_FILE_RENAMES.items():
        src = agentic / old
        dst = agentic / new
        if src.is_file() and not dst.exists():
            src.rename(dst)
            print(f"  agentic/{old} -> {new}")


def rename_sampling_folder() -> None:
    src = ROOT / "diffusion" / "sampling_extras"
    dst = ROOT / "diffusion" / SAMPLING_RENAME
    if src.is_dir() and not dst.exists():
        src.rename(dst)
        print("diffusion/sampling_extras -> diffusion/sampling")


def rename_test_files() -> None:
    tests = ROOT / "tests"
    for old, new in TEST_RENAMES.items():
        src = tests / old
        dst = tests / new
        if src.is_file() and not dst.exists():
            src.rename(dst)
            print(f"tests/{old} -> {new}")


def patch_text_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    orig = text
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_tree() -> int:
    exts = {".py", ".md", ".txt", ".ps1", ".yml", ".yaml", ".toml", ".cu", ".h", ".rs", ".go"}
    n = 0
    for path in ROOT.rglob("*"):
        if _should_skip(path):
            continue
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix.lower() not in exts and path.name not in {"README", "Dockerfile"}:
            continue
        if "rename_project_layout.py" in str(path) or "rename_advanced_innovations.py" in str(path):
            continue
        if patch_text_file(path):
            n += 1
            print(f"patched {path.relative_to(ROOT)}")
    return n


def write_sampling_extras_shim() -> None:
    shim_dir = ROOT / "diffusion" / "sampling_extras"
    if shim_dir.exists():
        return
    shim_dir.mkdir(parents=True)
    init = shim_dir / "__init__.py"
    init.write_text(
        '''"""
Deprecated import path — use ``diffusion.sampling``.

Re-exports lazily from the canonical package.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_CANONICAL = "diffusion.sampling"


def __getattr__(name: str) -> Any:
    mod = import_module(_CANONICAL)
    if not hasattr(mod, name):
        raise AttributeError(f"module {_CANONICAL!r} has no attribute {name!r}")
    val = getattr(mod, name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:
    return sorted(dir(import_module(_CANONICAL)))
''',
        encoding="utf-8",
    )
    readme = shim_dir / "README.md"
    readme.write_text(
        "# Deprecated\n\nUse [`diffusion/sampling`](../sampling/README.md) (`diffusion.sampling`).\n",
        encoding="utf-8",
    )
    print("created diffusion/sampling_extras shim")


def write_holy_grail_shim() -> None:
    init = ROOT / "diffusion" / "holy_grail" / "__init__.py"
    if not init.is_file():
        return
    text = init.read_text(encoding="utf-8")
    text = text.replace("diffusion.sampling_extras", "diffusion.sampling")
    text = text.replace("``diffusion.sampling_extras``", "``diffusion.sampling``")
    init.write_text(text, encoding="utf-8")
    readme = ROOT / "diffusion" / "holy_grail" / "README.md"
    if readme.is_file():
        t = readme.read_text(encoding="utf-8")
        t = t.replace("diffusion.sampling_extras", "diffusion.sampling")
        t = t.replace("../sampling_extras/", "../sampling/")
        readme.write_text(t, encoding="utf-8")
    print("updated diffusion/holy_grail shim -> sampling")


def write_innovations_compat_shim() -> None:
    """No-op: canonical package is ``innovations`` (no ``advanced_innovations`` shim)."""
    print("skip advanced_innovations compat shim (use innovations)")


def fix_innovations_init_exports() -> None:
    init = ROOT / "innovations" / "__init__.py"
    if not init.is_file():
        return
    text = init.read_text(encoding="utf-8")
    text = text.replace("from .integration import", "from .pipeline import")
    text = text.replace('"integration"', '"pipeline"')
    init.write_text(text, encoding="utf-8")


def fix_registry() -> None:
    reg = ROOT / "innovations" / "registry.py"
    if not reg.is_file():
        return
    text = reg.read_text(encoding="utf-8")
    text = text.replace("advanced_innovations.", "innovations.")
    text = text.replace("innovations.integration", "innovations.pipeline")
    reg.write_text(text, encoding="utf-8")


def fix_naming_compat_test() -> None:
    path = ROOT / "tests" / "test_naming_compat.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    # After bulk replace, tests may reference sampling; ensure compat tests still valid
    if "diffusion.sampling import list_holy_grail_presets as list_se" not in text:
        text = text.replace(
            "from diffusion.sampling import list_holy_grail_presets  # noqa: PLC0415",
            "from diffusion.holy_grail import list_holy_grail_presets  # noqa: PLC0415",
        )
        text = text.replace(
            "import diffusion.sampling as se  # noqa: PLC0415",
            "import diffusion.sampling as se  # noqa: PLC0415",
        )
    text = text.replace(
        "def test_diffusion_holy_grail_shim_reexports_sampling_extras():",
        "def test_diffusion_holy_grail_shim_reexports_sampling():",
    )
    text = text.replace(
        "def test_diffusion_holy_grail_all_matches_sampling_extras():",
        "def test_diffusion_holy_grail_all_matches_sampling():",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rename_top_level_package()
    pkg = ROOT / "innovations"
    rename_innovations_files(pkg)
    rename_agentic_files(pkg / "agentic")
    rename_sampling_folder()
    rename_test_files()
    n = patch_tree()
    print(f"patched {n} text files")
    write_sampling_extras_shim()
    write_holy_grail_shim()
    write_innovations_compat_shim()
    fix_innovations_init_exports()
    fix_registry()
    fix_naming_compat_test()


if __name__ == "__main__":
    main()
