"""
**Quality flywheel** orchestrator: curate manifest → auto-improve loop → promote best checkpoint.

Does not run full ``train.py`` (too project-specific); focuses on data curation + alignment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from config.defaults.superior_stack import FlywheelPlan

from .auto_loop import AutoImproveConfig, run_auto_improve


def curate_manifest(plan: FlywheelPlan, *, repo_root: Path, dry_run: bool = False) -> Optional[str]:
    """Run ``superior_curate`` when ``manifest_in`` and ``manifest_out`` are set."""
    if plan.skip_curate or not plan.manifest_in or not plan.manifest_out:
        return plan.manifest_in
    d = plan.defaults
    cmd = [
        sys.executable,
        "-m",
        "scripts.tools",
        "superior_curate",
        plan.manifest_in,
        "--out",
        plan.manifest_out,
        "--dedup",
        d.dedup,
        "--min-caption-len",
        str(d.min_caption_len),
        "--max-caption-len",
        str(d.max_caption_len),
    ]
    if dry_run:
        print(" ".join(cmd))
        return plan.manifest_out
    rc = subprocess.run(cmd, cwd=str(repo_root), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"superior_curate failed with exit {rc}")
    return plan.manifest_out


def run_flywheel(
    plan: FlywheelPlan,
    *,
    repo_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute flywheel steps. Returns summary dict written to ``work_dir/flywheel_summary.json``.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    work = Path(plan.work_dir)
    if not dry_run:
        work.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"plan": _plan_to_dict(plan), "steps": []}

    if not plan.skip_curate and plan.manifest_in and plan.manifest_out:
        out = curate_manifest(plan, repo_root=root, dry_run=dry_run)
        summary["steps"].append({"curate": out})

    if not plan.skip_align:
        d = plan.defaults
        rag_jsonl = plan.local_rag_jsonl or ""
        cfg = AutoImproveConfig(
            base_ckpt=plan.base_ckpt,
            work_dir=str(work / "align"),
            preset=d.preset,
            suite_pack=d.benchmark_suite_pack,
            num=d.num_candidates,
            pick_best=d.pick_best,
            steps=d.steps,
            iterations=d.auto_loop_iterations,
            dpo_steps=d.dpo_steps,
            dpo_beta=d.dpo_beta,
            vit_ckpt=plan.vit_ckpt or "",
            local_rag_jsonl=rag_jsonl or "",
            model_soup=d.model_soup,
            soup_weight_base=d.soup_weight_base,
            soup_weight_dpo=d.soup_weight_dpo,
            promote_best=True,
            promote_path=plan.promote_path,
            extra_args=list(plan.extra_auto_loop_args),
        )
        rc = run_auto_improve(cfg, repo_root=root, dry_run=dry_run)
        summary["steps"].append({"align_exit_code": rc, "promote_path": plan.promote_path})
        if rc != 0 and not dry_run:
            summary["status"] = "align_failed"
            _write_summary(work, summary, dry_run)
            return summary

    summary["status"] = "ok"
    summary["best_ckpt"] = plan.promote_path
    _write_summary(work, summary, dry_run)
    return summary


def _plan_to_dict(plan: FlywheelPlan) -> Dict[str, Any]:
    d = asdict(plan)
    d["defaults"] = asdict(plan.defaults)
    return d


def _write_summary(work: Path, summary: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(summary, indent=2))
        return
    (work / "flywheel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = ["curate_manifest", "run_flywheel"]
