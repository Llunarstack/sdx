"""Keyframe editor: invoke sample.py on selected frames."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .types import KeyframeEditJob

__all__ = ["build_sample_cmd_for_keyframe", "run_keyframe_edit", "run_keyframe_batch"]


def build_sample_cmd_for_keyframe(
    job: KeyframeEditJob,
    *,
    ckpt: str,
    repo_root: Optional[Path] = None,
    image_size: int = 512,
    steps: int = 20,
    cfg_scale: float = 6.5,
    seed: int = 42,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    root = repo_root or Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(root / "sample.py"),
        "--ckpt",
        ckpt,
        "--prompt",
        job.prompt,
        "--negative-prompt",
        job.negative or " ",
        "--init-image",
        job.source_frame_path,
        "--out",
        job.output_path,
        "--image-size",
        str(int(image_size)),
        "--steps",
        str(int(steps)),
        "--cfg-scale",
        str(float(cfg_scale)),
        "--seed",
        str(int(seed) + job.frame_index),
    ]
    strength = float(job.init_strength)
    if strength > 0:
        cmd.extend(["--strength", str(strength)])
    if job.mask_path and Path(job.mask_path).is_file():
        cmd.extend(["--mask", job.mask_path, "--inpaint-mode", "mdm"])
    if extra_args:
        cmd.extend(list(extra_args))
    if job.sample_extra_args:
        for i, tok in enumerate(job.sample_extra_args):
            if tok.startswith("-") and tok in cmd:
                continue
            cmd.append(tok)
    return cmd


def run_keyframe_edit(
    job: KeyframeEditJob,
    *,
    ckpt: str,
    dry_run: bool = False,
    **kwargs: Any,
) -> Path:
    Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = build_sample_cmd_for_keyframe(job, ckpt=ckpt, **kwargs)
    if dry_run:
        return Path(job.output_path)
    subprocess.run(cmd, check=True)
    return Path(job.output_path)


def run_keyframe_batch(
    jobs: Sequence[KeyframeEditJob],
    *,
    ckpt: str,
    dry_run: bool = False,
    **kwargs: Any,
) -> List[Path]:
    outs: List[Path] = []
    for job in jobs:
        outs.append(run_keyframe_edit(job, ckpt=ckpt, dry_run=dry_run, **kwargs))
    return outs
