"""
Python API for the full **auto-improve** loop (benchmark → mine → DPO → re-benchmark → soup).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(slots=True)
class AutoImproveConfig:
    base_ckpt: str
    work_dir: str = "auto_improve_loop"
    preset: str = "sdxl"
    device: str = "cuda"
    suite_pack: str = "top_contender_proxy_v1"
    steps: int = 30
    num: int = 4
    pick_best: str = "superior_composite"
    iterations: int = 1
    dpo_steps: int = 500
    dpo_beta: float = 300.0
    vit_ckpt: str = ""
    local_rag_jsonl: str = ""
    model_soup: bool = False
    soup_weight_base: float = 0.35
    soup_weight_dpo: float = 0.65
    promote_best: bool = True
    promote_path: str = "auto_improve_loop/best_auto.pt"
    extra_args: List[str] = field(default_factory=list)


def build_auto_improve_argv(cfg: AutoImproveConfig) -> List[str]:
    argv = [
        "--base-ckpt",
        cfg.base_ckpt,
        "--work-dir",
        cfg.work_dir,
        "--preset",
        cfg.preset,
        "--device",
        cfg.device,
        "--suite-pack",
        cfg.suite_pack,
        "--steps",
        str(cfg.steps),
        "--num",
        str(cfg.num),
        "--pick-best",
        cfg.pick_best,
        "--iterations",
        str(cfg.iterations),
        "--dpo-steps",
        str(cfg.dpo_steps),
        "--dpo-beta",
        str(cfg.dpo_beta),
    ]
    if cfg.vit_ckpt:
        argv.extend(["--vit-ckpt", cfg.vit_ckpt])
    if cfg.local_rag_jsonl:
        argv.extend(["--local-rag-jsonl", cfg.local_rag_jsonl])
    if cfg.model_soup:
        argv.append("--model-soup")
        argv.extend(["--soup-weight-base", str(cfg.soup_weight_base)])
        argv.extend(["--soup-weight-dpo", str(cfg.soup_weight_dpo)])
    if cfg.promote_best:
        argv.append("--promote-best")
        argv.extend(["--promote-path", cfg.promote_path])
    argv.extend(cfg.extra_args)
    return argv


def run_auto_improve(
    cfg: AutoImproveConfig,
    *,
    repo_root: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    cmd = [sys.executable, "-m", "scripts.tools", "auto_improve_loop", *build_auto_improve_argv(cfg)]
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(root), check=False).returncode


__all__ = ["AutoImproveConfig", "build_auto_improve_argv", "run_auto_improve"]
