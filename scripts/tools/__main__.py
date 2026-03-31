"""
Run canonical tool scripts without duplicate flat wrappers::

    python -m scripts.tools ckpt_info results/run/best.pt
    python -m scripts.tools data_quality --help
    python -m scripts.tools update_project_structure --max-depth 4

Commands accept underscores or hyphens (e.g. ``ckpt-info``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# command name -> path to canonical script (under scripts/tools/)
_CANONICAL: dict[str, Path] = {
    "ckpt_info": _HERE / "dev" / "ckpt_info.py",
    "smoke_imports": _HERE / "dev" / "smoke_imports.py",
    "quick_test": _HERE / "dev" / "quick_test.py",
    "ar_mask_inspect": _HERE / "dev" / "ar_mask_inspect.py",
    "noise_schedule_export": _HERE / "training" / "noise_schedule_export.py",
    "data_quality": _HERE / "data" / "data_quality.py",
    "manifest_paths": _HERE / "data" / "manifest_paths.py",
    "jsonl_merge": _HERE / "data" / "jsonl_merge.py",
    "prompt_lint": _HERE / "prompt" / "prompt_lint.py",
    "tag_coverage": _HERE / "prompt" / "tag_coverage.py",
    "export_onnx": _HERE / "export" / "export_onnx.py",
    "export_safetensors": _HERE / "export" / "export_safetensors.py",
    "op_preflight": _HERE / "ops" / "op_preflight.py",
    "orchestrate_pipeline": _HERE / "ops" / "orchestrate_pipeline.py",
    "update_project_structure": _HERE / "repo" / "update_project_structure.py",
    "verify_doc_links": _HERE / "repo" / "verify_doc_links.py",
    "clean_repo_artifacts": _HERE / "repo" / "clean_repo_artifacts.py",
}


def _commands_help() -> str:
    names = sorted(_CANONICAL.keys())
    return "  " + "\n  ".join(names)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__ or "")
        print("Commands:\n" + _commands_help())
        return 0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help") else 1

    raw = sys.argv[1]
    cmd = raw.replace("-", "_")
    target = _CANONICAL.get(cmd)
    if target is None or not target.is_file():
        print(f"Unknown command: {raw!r}\n\n" + _commands_help(), file=sys.stderr)
        return 2

    # Delegate: script sees argv[0] as this path; remaining args unchanged.
    sys.argv = [str(target.resolve())] + sys.argv[2:]
    runpy.run_path(str(target.resolve()), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
