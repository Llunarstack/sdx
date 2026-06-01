"""
Smoke test: import all *internal* SDX modules to catch broken links early.

This does NOT try to run training or load huge checkpoints; it only checks that imports
across the codebase succeed (excluding `external/`).

Usage:
    python -m scripts.tools smoke_imports
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path


def iter_internal_modules(root_pkg: str, root_path: Path):
    for m in pkgutil.walk_packages([str(root_path)], prefix=f"{root_pkg}."):
        yield m.name


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    # Only include internal packages we own.
    packages = [
        ("config", repo_root / "config"),
        ("data", repo_root / "data"),
        ("diffusion", repo_root / "diffusion"),
        ("models", repo_root / "models"),
        ("training", repo_root / "training"),
        ("utils", repo_root / "utils"),
        ("pipelines", repo_root / "pipelines"),
        ("vit_quality", repo_root / "vit_quality"),
        ("scripts.tools", repo_root / "scripts" / "tools"),
    ]

    skip_prefixes = (
        "research.agi_image.",  # experimental scaffold; optional heavy deps
        "training.enhanced_trainer",  # depends on archived utils.architecture
        "training.train_args",  # depends on archived utils.modeling
        "training.book_train_preset",  # depends on sdx_native
        "utils.generation.cfg_batched",  # depends on archived utils.superior
        "utils.generation.dit_ar_latent_compat",  # depends on archived utils.architecture
        "utils.generation.guidance_probe",  # depends on archived utils.superior
        "utils.generation.guidance_stack",  # depends on archived utils.superior
        "utils.generation.master_integration",  # depends on archived utils.analysis
        "utils.generation.multimodal_generation",  # depends on archived utils.architecture
        "utils.generation.zeresfdg",  # depends on archived utils.superior
        "utils.modeling",  # depends on sdx_native and has circular imports
        "utils.prompt.scene_blueprint",  # depends on archived utils.runtime
        "utils.prompt.shape_scaffold",  # depends on archived utils.runtime
        "utils.runtime",  # depends on archived submodules
        "vit_quality.checkpoint_utils",  # depends on archived utils.architecture
        "vit_quality.dataset",  # depends on archived utils.architecture
        "vit_quality.export_embeddings",  # depends on archived utils.runtime
        "vit_quality.infer",  # depends on archived utils.runtime
        "vit_quality.model",  # depends on archived utils.architecture
        "vit_quality.prompt_tool",  # depends on archived utils.runtime
        "vit_quality.rank",  # depends on archived utils.runtime
        "vit_quality.train",  # depends on archived utils.runtime
        "pipelines.book_comic.book_model_readiness",  # depends on archived utils.architecture
        "pipelines.book_comic.book_training_helpers",  # depends on sdx_native
        "scripts.tools.data.caption_hygiene",  # depends on sdx_native
        "scripts.tools.data.manifest_paths",  # depends on sdx_native
        "scripts.tools.dev.architecture_themes",  # depends on archived utils.architecture
        "scripts.tools.ops.agentic_flywheel",  # depends on archived utils.agentic
        "scripts.tools.ops.agentic_roles",  # depends on archived utils.agentic
        "scripts.tools.ops.agentic_evolve",  # depends on archived utils.agentic
        "scripts.tools.ops.agentic_generate",  # depends on archived utils.agentic
        "scripts.tools.ops.gen_searcher_bridge",  # depends on archived utils.modeling
        "scripts.tools.ops.pretrained_status",  # depends on archived utils.modeling
        "scripts.tools.ops.visual_brain_generate",  # depends on archived utils.agentic
    )

    failures = []
    for pkg, path in packages:
        if not path.exists():
            continue
        for mod_name in iter_internal_modules(pkg, path):
            if mod_name.endswith(".__pycache__"):
                continue
            if any(mod_name.startswith(p) for p in skip_prefixes):
                continue
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failures.append((mod_name, repr(e)))

    if failures:
        print("SMOKE IMPORTS: failures:")
        for mod_name, err in failures:
            print(f"- {mod_name}: {err}")
        raise SystemExit(1)

    print("SMOKE IMPORTS: all internal modules imported successfully.")


if __name__ == "__main__":
    main()
