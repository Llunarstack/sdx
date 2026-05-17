"""
Run **img2img** and **inpainting** through the same path as ``sample.py`` (subprocess).

This keeps API callers (e.g. :class:`MultimodalGenerator`) decoupled from loading VAE/DiT
in-process. Requires a valid checkpoint and a working SDX environment (same as CLI sampling).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError("sample_edit_runner requires Pillow") from e

__all__ = [
    "build_sample_command",
    "resolve_repo_root",
    "resolve_sample_py",
    "run_edit_with_pillow",
    "run_sample_inference",
]


def resolve_repo_root() -> Path:
    """SDX repository root (parent of ``sample.py``)."""
    return Path(__file__).resolve().parents[2]


def resolve_sample_py(repo_root: Optional[Path] = None) -> Path:
    """Absolute path to the repository ``sample.py`` entrypoint."""
    root = repo_root if repo_root is not None else resolve_repo_root()
    p = root / "sample.py"
    if not p.is_file():
        raise FileNotFoundError(f"sample.py not found at {p} (unexpected layout?)")
    return p


def build_sample_command(
    *,
    ckpt: str,
    prompt: str,
    out_path: Union[str, Path],
    repo_root: Optional[Path] = None,
    sample_py: Optional[Path] = None,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: Optional[int] = None,
    num: int = 1,
    device: str = "cuda",
    init_image_path: Optional[Union[str, Path]] = None,
    mask_image_path: Optional[Union[str, Path]] = None,
    strength: float = 0.65,
    inpaint_mode: str = "mdm",
    scheduler: str = "ddim",
    solver: str = "ddim",
    timestep_schedule: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Construct a ``sample.py`` argv list (executable + args). Caller runs it with no shell.

    Raises:
        ValueError: invalid dimensions, steps, ``strength``, or mask without init image.
    """
    if int(width) < 1 or int(height) < 1:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    sf = float(strength)
    if not 0.0 <= sf <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {sf}")

    root = repo_root if repo_root is not None else resolve_repo_root()
    script = sample_py if sample_py is not None else resolve_sample_py(root)

    if mask_image_path and not init_image_path:
        raise ValueError("Inpainting requires both init image and mask paths.")

    cmd: List[str] = [
        sys.executable,
        str(script),
        "--ckpt",
        ckpt,
        "--prompt",
        prompt,
        "--out",
        str(out_path),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--steps",
        str(int(steps)),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--num",
        str(int(max(1, num))),
        "--device",
        device,
        "--scheduler",
        scheduler,
        "--solver",
        solver,
    ]
    ts = (timestep_schedule or "").strip()
    if ts:
        cmd.extend(["--timestep-schedule", ts])
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if negative_prompt.strip():
        cmd.extend(["--negative-prompt", negative_prompt])

    init_p = Path(init_image_path) if init_image_path else None
    if init_p is not None:
        cmd.extend(["--init-image", str(init_p), "--strength", str(float(strength))])

    mask_p = Path(mask_image_path) if mask_image_path else None
    if mask_p is not None:
        cmd.extend(["--mask", str(mask_p), "--inpaint-mode", str(inpaint_mode or "mdm")])

    if extra_args:
        for a in extra_args:
            if not isinstance(a, str) or not a.strip():
                raise ValueError(f"extra_args entries must be non-empty strings, got {a!r}")
        cmd.extend(list(extra_args))

    return cmd


def run_sample_inference(
    *,
    ckpt: str,
    prompt: str,
    out_path: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None,
    check: bool = True,
    capture_output: bool = True,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: Optional[int] = None,
    num: int = 1,
    device: str = "cuda",
    init_image_path: Optional[Union[str, Path]] = None,
    mask_image_path: Optional[Union[str, Path]] = None,
    strength: float = 0.65,
    inpaint_mode: str = "mdm",
    scheduler: str = "ddim",
    solver: str = "ddim",
    timestep_schedule: Optional[str] = None,
    repo_root: Optional[Path] = None,
    sample_py: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run ``sample.py`` subprocess.

    ``cwd`` defaults to repository root so relative checkpoint paths behave like the CLI.
    """
    root = resolve_repo_root()
    cw = cwd if cwd is not None else root

    cmd = build_sample_command(
        ckpt=ckpt,
        prompt=prompt,
        out_path=out_path,
        repo_root=repo_root,
        sample_py=sample_py,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        num=num,
        device=device,
        init_image_path=init_image_path,
        mask_image_path=mask_image_path,
        strength=strength,
        inpaint_mode=inpaint_mode,
        scheduler=scheduler,
        solver=solver,
        timestep_schedule=timestep_schedule,
        extra_args=extra_args,
    )

    result = subprocess.run(
        cmd,
        cwd=str(cw),
        capture_output=capture_output,
        text=True,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        tail = (stderr + "\n" + stdout)[-4000:]
        raise RuntimeError(f"sample.py failed ({result.returncode}): {tail}")
    return result


def run_edit_with_pillow(
    *,
    ckpt: str,
    prompt: str,
    negative_prompt: str,
    base_image: Image.Image,
    mask_image: Optional[Image.Image],
    width: int,
    height: int,
    steps: int = 28,
    cfg_scale: float = 7.0,
    seed: Optional[int] = None,
    img2img_strength: float = 0.65,
    inpaint_mode: str = "mdm",
    device: str = "cuda",
    scheduler: str = "ddim",
    solver: str = "ddim",
    timestep_schedule: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    keep_temp_dir: bool = False,
) -> Image.Image:
    """
    Save ``base_image`` (+ optional grayscale ``mask_image``) to temp files and run inpaint/img2img.

    Mask convention matches ``sample.py``: **white** = inpaint / regenerate region.
    """
    base_rgb = base_image.convert("RGB").resize((int(width), int(height)), Image.Resampling.LANCZOS)
    mask_l: Optional[Image.Image] = None
    if mask_image is not None:
        mask_l = mask_image.convert("L").resize((int(width), int(height)), Image.Resampling.LANCZOS)

    root = resolve_repo_root()
    tmp_home = tempfile.mkdtemp(prefix="sdx_edit_")
    tmp = Path(tmp_home)
    try:
        init_path = tmp / "init.png"
        out_path = tmp / "out.png"
        base_rgb.save(init_path, format="PNG")
        mask_path: Optional[Path] = None
        if mask_l is not None:
            mask_path = tmp / "mask.png"
            mask_l.save(mask_path, format="PNG")

        run_sample_inference(
            ckpt=ckpt,
            prompt=prompt,
            out_path=out_path,
            cwd=root,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            init_image_path=init_path,
            mask_image_path=mask_path,
            strength=img2img_strength,
            inpaint_mode=inpaint_mode,
            device=device,
            scheduler=scheduler,
            solver=solver,
            timestep_schedule=timestep_schedule,
            extra_args=extra_args,
        )
        if not out_path.is_file():
            raise RuntimeError(f"sample.py reported success but output missing: {out_path}")
        return Image.open(out_path).convert("RGB")
    finally:
        if not keep_temp_dir:
            shutil.rmtree(tmp_home, ignore_errors=True)
