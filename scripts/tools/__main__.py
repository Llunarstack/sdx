"""
Run any tool script under ``scripts/tools/`` by its file name::

    python -m scripts.tools ckpt_info results/run/best.pt
    python -m scripts.tools data_quality --help
    python -m scripts.tools update_project_structure --max-depth 4

Every ``scripts/tools/**/<name>.py`` with a ``__main__`` guard is a command; names accept underscores
or hyphens (e.g. ``ckpt-info``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from scripts.tools._repo_bootstrap import ensure_repo_paths

ensure_repo_paths()

from utils.terminal import configure_stdio_for_console  # noqa: E402

# Tool scripts don't all guard their own stdio; without this, argparse help
# containing arrows crashes on legacy Windows console encodings (cp1252).
configure_stdio_for_console()

_HERE = Path(__file__).resolve().parent
# alias -> "script [subcommand ...]"; merged scripts keep their old command names here.
_ALIASES = {
    "feedback": "feedback_cli",
    "visual_brain": "visual_brain_generate",
    "corpus_retrieve": "corpus_retrieve_cli",
    "superior_generate": "superior generate",
    "superior_ensemble": "superior ensemble",
    "superior_curate": "superior curate",
    "superior_eval_report": "superior report",
    "superior_dpo_loop": "superior dpo",
    "superior_auto_loop": "superior auto-loop",
    "run_flywheel": "superior flywheel",
    "agentic_generate": "agentic generate",
    "agentic_roles": "agentic roles",
    "agentic_evolve": "agentic evolve",
    "agentic_flywheel": "agentic flywheel",
}


def commands() -> dict[str, tuple[Path, list[str]]]:
    """Command name -> (script path, leading args): file stems plus ``_ALIASES``."""
    out: dict[str, tuple[Path, list[str]]] = {
        p.stem: (p, [])
        for p in sorted(_HERE.rglob("*.py"))
        if not p.name.startswith("_")
        and "__pycache__" not in p.parts
        and 'if __name__ == "__main__"' in p.read_text(encoding="utf-8", errors="ignore")
    }
    for alias, spec in _ALIASES.items():
        target, *lead = spec.split()
        if target in out:
            out[alias] = (out[target][0], lead)
    return out


def main() -> int:
    cmds = commands()
    listing = "Commands:\n  " + "\n  ".join(sorted(cmds))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__ or "")
        print(listing)
        return 0 if len(sys.argv) >= 2 else 1

    raw = sys.argv[1]
    found = cmds.get(raw.replace("-", "_"))
    if found is None:
        print(f"Unknown command: {raw!r}\n\n{listing}", file=sys.stderr)
        return 2

    # The script sees argv[0] as its own path, then any alias subcommand, then the user's args.
    target, lead = found
    sys.argv = [str(target), *lead, *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
