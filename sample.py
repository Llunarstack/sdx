"""
Generate an image from a text prompt with a trained checkpoint (``python sample.py --help`` for every option).

The implementation lives in ``utils.generation.sample_main``; ``--preset`` applies a model-style preset from
``config.defaults.model_presets``. ``--profile-out PATH`` (with ``--profile-sort`` / ``--profile-top``) writes a
cProfile ``.prof`` and a text summary next to PATH.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.terminal import configure_stdio_for_console

configure_stdio_for_console()

# ``--help`` loads the argparse-only parser by path so the utils.generation package (and torch) is not imported.
if __name__ == "__main__" and any(_h in sys.argv[1:] for _h in ("-h", "--help")):
    _spec = _ilu.spec_from_file_location(
        "_sdx_sample_cli_parser",
        Path(__file__).resolve().parent / "utils" / "generation" / "sample_cli_parser.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    _mod.build_sample_parser().parse_args()
    raise SystemExit(0)


def build_sample_parser():
    from utils.generation.sample_cli_parser import build_sample_parser as _build

    return _build()


def load_model_from_ckpt(ckpt_path, device="cuda"):
    from utils.generation.sample_helpers import load_model_from_ckpt as _load

    return _load(ckpt_path, device=device)


def encode_text(*args, **kwargs):
    from utils.generation.sample_helpers import encode_text as _encode

    return _encode(*args, **kwargs)


def main():
    from utils.generation.sample_main import main as _main

    return _main()


if __name__ == "__main__":
    from utils.runtime import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv
    if _pcfg is not None:
        run_with_cprofile(main, _pcfg)
    else:
        main()
