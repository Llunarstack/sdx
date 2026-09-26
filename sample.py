"""
Generate an image from a text prompt using a trained checkpoint.
Supports: prompt, negative prompt, steps, width, height, CFG, timestep schedules (ddim, euler, karras_rho, …) and solvers (ddim, heun).
Optional: style, control-image, lora, img2img, inpainting, dual-stage layout (coarse latent then detail pass),
hires-fix (latent upscale + refine), volatile CFG (spike-aware guidance), CLIP-guard extra denoise,
CLIP monitor (mid-loop CFG boost on low cosine), spectral-coherence latent (FFT lowfreq blend),
domain latent prior, sharpen, contrast, saturation, clarity / tone-punch / chroma-smooth / polish /
finishing-preset (cross-style post), emphasis (word)/[word].

Presets and OP modes:
- --preset sdxl|flux|anime|zit: apply a model-style preset from config.defaults.model_presets.
- --op-mode portrait|fullbody|anime_char: apply a high-level OP bundle on top.

Profiling (optional): pass ``--profile-out PATH`` (plus ``--profile-sort cumulative|tottime|...``,
``--profile-top N``) to write cProfile ``.prof`` and a text summary next to PATH.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.terminal import configure_stdio_for_console

configure_stdio_for_console()

# Show CLI help without importing the heavy GPU stack: load the argparse-only
# parser by path so we do not trigger utils.generation package side imports.
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
    from utils.runtime.profiling import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv
    if _pcfg is not None:
        run_with_cprofile(main, _pcfg)
    else:
        main()
