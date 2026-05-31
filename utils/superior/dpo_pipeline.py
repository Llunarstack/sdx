"""
End-to-end **DPO alignment** helpers: mine preferences → train policy → export checkpoint.

Wraps existing ``mine_preference_pairs``, ``train_diffusion_dpo``, and ``PreferenceImageDataset``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass(slots=True)
class DPOStageConfig:
    ckpt: str
    preference_jsonl: str
    image_root: str = ""
    out: str = "dpo_policy.pt"
    steps: int = 500
    batch_size: int = 2
    lr: float = 1e-6
    dpo_beta: float = 300.0
    dpo_logit_clip: float = 40.0
    sync_ref_every: int = 0
    ref_ema_alpha: float = 0.01
    timestep_dpo_weight: str = "high_noise"
    timestep_dpo_power: float = 0.5
    safeguarded_dpo: float = 0.85
    device: str = "cuda"


@dataclass(slots=True)
class MinePairsConfig:
    benchmark_json: str
    out_jsonl: str
    min_margin: float = 0.08
    max_pairs_per_case: int = 2


def mine_pairs_from_benchmark(cfg: MinePairsConfig) -> List[Dict[str, Any]]:
    """Load benchmark results JSON and mine win/lose pairs."""
    import importlib.util

    mod_path = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "training" / "mine_preference_pairs.py"
    spec = importlib.util.spec_from_file_location("mine_preference_pairs", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mine_pairs = mod.mine_pairs

    path = Path(cfg.benchmark_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("results", payload.get("rows", []))
    pairs = mine_pairs(rows, min_margin=cfg.min_margin, max_pairs_per_case=cfg.max_pairs_per_case)
    out = Path(cfg.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return pairs


def build_dpo_train_argv(cfg: DPOStageConfig) -> List[str]:
    """Argv for ``python -m scripts.tools train_diffusion_dpo``."""
    return [
        "--ckpt",
        cfg.ckpt,
        "--preference-jsonl",
        cfg.preference_jsonl,
        "--image-root",
        cfg.image_root,
        "--out",
        cfg.out,
        "--steps",
        str(cfg.steps),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--dpo-beta",
        str(cfg.dpo_beta),
        "--dpo-logit-clip",
        str(cfg.dpo_logit_clip),
        "--sync-ref-every",
        str(cfg.sync_ref_every),
        "--ref-ema-alpha",
        str(cfg.ref_ema_alpha),
        "--timestep-dpo-weight",
        str(cfg.timestep_dpo_weight),
        "--timestep-dpo-power",
        str(cfg.timestep_dpo_power),
        "--safeguarded-dpo",
        str(cfg.safeguarded_dpo),
        "--device",
        cfg.device,
    ]


def run_dpo_training(
    cfg: DPOStageConfig,
    *,
    repo_root: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
) -> int:
    """Run DPO trainer subprocess."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    cmd = [sys.executable, "-m", "scripts.tools", "train_diffusion_dpo", *build_dpo_train_argv(cfg)]
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(root), check=False).returncode


def run_alignment_loop(
    *,
    benchmark_json: str,
    base_ckpt: str,
    pairs_jsonl: str,
    dpo_out: str,
    mine: Optional[MinePairsConfig] = None,
    dpo: Optional[DPOStageConfig] = None,
    repo_root: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Mine pairs from benchmark → train DPO policy. Returns summary dict.
    """
    mine_cfg = mine or MinePairsConfig(benchmark_json=benchmark_json, out_jsonl=pairs_jsonl)
    mine_cfg.benchmark_json = benchmark_json
    mine_cfg.out_jsonl = pairs_jsonl
    pairs = mine_pairs_from_benchmark(mine_cfg)
    dpo_cfg = dpo or DPOStageConfig(ckpt=base_ckpt, preference_jsonl=pairs_jsonl, out=dpo_out)
    dpo_cfg.ckpt = base_ckpt
    dpo_cfg.preference_jsonl = pairs_jsonl
    dpo_cfg.out = dpo_out
    rc = run_dpo_training(dpo_cfg, repo_root=repo_root, dry_run=dry_run)
    return {"pairs_mined": len(pairs), "dpo_exit_code": rc, "dpo_out": dpo_out}


__all__ = [
    "DPOStageConfig",
    "MinePairsConfig",
    "build_dpo_train_argv",
    "mine_pairs_from_benchmark",
    "run_alignment_loop",
    "run_dpo_training",
]
