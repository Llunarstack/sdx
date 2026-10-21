"""Python API for the full **auto-improve** loop (benchmark → mine → DPO → re-benchmark → soup).

Also holds what used to be: dpo_pipeline, flywheel, hard_negative, vit_mining, model_soup, eval_report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from config.defaults._lexicon import merge_csv_unique
from config.defaults.superior_stack import FlywheelPlan
from PIL import Image

from utils.modeling.model_paths import repo_root as _sdx_repo_root

# ==================================================================================================
# auto_loop
# ==================================================================================================


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
    extra_args: list[str] = field(default_factory=list)


def build_auto_improve_argv(cfg: AutoImproveConfig) -> list[str]:
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


def run_tool(tool: str, argv: list[str], *, repo_root: str | Path | None = None, dry_run: bool = False) -> int:
    """Run ``python -m scripts.tools <tool> <argv>`` from the repo root; ``dry_run`` only prints it."""
    cmd = [sys.executable, "-m", "scripts.tools", tool, *argv]
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(repo_root or _sdx_repo_root()), check=False).returncode


def run_auto_improve(cfg: AutoImproveConfig, *, repo_root: Path | None = None, dry_run: bool = False) -> int:
    return run_tool("auto_improve_loop", build_auto_improve_argv(cfg), repo_root=repo_root, dry_run=dry_run)


# ==================================================================================================
# dpo_pipeline: End-to-end **DPO alignment** helpers: mine preferences → train policy → export checkpoint.
#
# Wraps existing ``mine_preference_pairs``, ``train_diffusion_dpo``, and ``PreferenceImageDataset``.
# ==================================================================================================


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


def mine_pairs_from_benchmark(cfg: MinePairsConfig) -> list[dict[str, Any]]:
    """Load benchmark results JSON and mine win/lose pairs."""
    import importlib.util

    mod_path = _sdx_repo_root() / "scripts" / "tools" / "training" / "mine_preference_pairs.py"
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


def build_dpo_train_argv(cfg: DPOStageConfig) -> list[str]:
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
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    return run_tool("train_diffusion_dpo", build_dpo_train_argv(cfg), repo_root=repo_root, dry_run=dry_run)


def run_alignment_loop(
    *,
    benchmark_json: str,
    base_ckpt: str,
    pairs_jsonl: str,
    dpo_out: str,
    mine: MinePairsConfig | None = None,
    dpo: DPOStageConfig | None = None,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
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


# ==================================================================================================
# flywheel: **Quality flywheel** orchestrator: curate manifest → auto-improve loop → promote best checkpoint.
#
# Does not run full ``train.py`` (too project-specific); focuses on data curation + alignment.
# ==================================================================================================


def curate_manifest(plan: FlywheelPlan, *, repo_root: Path, dry_run: bool = False) -> str | None:
    """Run ``superior_curate`` when ``manifest_in`` and ``manifest_out`` are set."""
    if plan.skip_curate or not plan.manifest_in or not plan.manifest_out:
        return plan.manifest_in
    d = plan.defaults
    argv = [
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
    rc = run_tool("superior_curate", argv, repo_root=repo_root, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"superior_curate failed with exit {rc}")
    return plan.manifest_out


def run_flywheel(
    plan: FlywheelPlan,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute flywheel steps. Returns summary dict written to ``work_dir/flywheel_summary.json``.
    """
    root = Path(repo_root or _sdx_repo_root())
    work = Path(plan.work_dir)
    if not dry_run:
        work.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"plan": _plan_to_dict(plan), "steps": []}

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


def _plan_to_dict(plan: FlywheelPlan) -> dict[str, Any]:
    d = asdict(plan)
    d["defaults"] = asdict(plan.defaults)
    return d


def _write_summary(work: Path, summary: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(summary, indent=2))
        return
    (work / "flywheel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ==================================================================================================
# hard_negative: Mine **hard negative prompt tokens** from failed benchmark / hard-case exports.
#
# Appends domain-specific negatives to reduce recurring failure modes (blur, text, counting).
# ==================================================================================================

_TAG_NEGATIVES: dict[str, str] = {
    "low_composite": "low quality, blurry, artifacts",
    "text_rendering": "misspelled text, illegible lettering, garbled typography",
    "counting": "wrong number of objects, extra limbs, duplicated subjects",
    "exposure": "overexposed, underexposed, blown highlights, crushed shadows",
    "oversaturation": "oversaturated, neon colors, unrealistic skin tones",
    "blur": "blurry, out of focus, soft details, motion blur",
}


@dataclass(slots=True)
class HardNegativeBundle:
    tags: list[str]
    negative_suffix: str
    per_tag_counts: dict[str, int]


def tags_from_benchmark_row(row: dict[str, Any], *, threshold: float = 0.60) -> list[str]:
    """Infer failure tags from a benchmark ``results.json`` row (mirrors benchmark_suite)."""
    tags: list[str] = []
    comp = float(row.get("composite", 0.0) or 0.0)
    if comp < threshold:
        tags.append("low_composite")
    if float(row.get("ocr_match", 1.0) or 1.0) < 0.7 and row.get("expected_text"):
        tags.append("text_rendering")
    if float(row.get("count_match", 1.0) or 1.0) < 0.7 and int(row.get("expected_count", 0) or 0) > 0:
        tags.append("counting")
    if float(row.get("exposure_balance", 0.5) or 0.5) < 0.6:
        tags.append("exposure")
    if float(row.get("saturation_balance", 0.5) or 0.5) < 0.6:
        tags.append("oversaturation")
    edge = float(row.get("edge_sharpness", 0.0) or 0.0)
    if edge / 400.0 < 0.35:
        tags.append("blur")
    return tags


def mine_hard_negatives(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.60,
    max_phrases: int = 12,
) -> HardNegativeBundle:
    """Aggregate failure tags and build a negative prompt suffix."""
    counter: Counter[str] = Counter()
    for row in rows:
        for t in tags_from_benchmark_row(row, threshold=threshold):
            counter[t] += 1
        for t in row.get("failure_tags") or []:
            counter[str(t)] += 1
    ordered = [t for t, _ in counter.most_common(max_phrases)]
    suffix = merge_csv_unique(*(_TAG_NEGATIVES.get(tag, "") for tag in ordered))
    return HardNegativeBundle(tags=ordered, negative_suffix=suffix, per_tag_counts=dict(counter))


def load_hard_negatives_from_results(
    path: str | Path,
    *,
    threshold: float = 0.60,
) -> HardNegativeBundle:
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = []
    low = [r for r in rows if isinstance(r, dict) and float(r.get("composite", 1.0) or 1.0) < threshold]
    return mine_hard_negatives(low or rows, threshold=threshold)


def benchmark_sample_args_for_negatives(bundle: HardNegativeBundle) -> list[str]:
    """Return ``benchmark_suite --sample-arg`` tokens for mined hard negatives."""
    if not bundle.negative_suffix.strip():
        return []
    return ["--negative-prompt", bundle.negative_suffix]


# ==================================================================================================
# vit_mining: **ViT-guided preference mining** — use ``vit_quality`` scores to rank candidates and mine DPO pairs.
#
# Falls back to benchmark ``composite`` when ViT checkpoint is unavailable.
# ==================================================================================================


@dataclass(slots=True)
class ViTMineConfig:
    vit_ckpt: str
    min_margin: float = 0.06
    max_pairs_per_case: int = 2
    vit_weight: float = 0.45
    composite_weight: float = 0.55
    device: str = "cuda"
    use_adherence: bool = True


def score_image_vit(
    image_path: str | Path,
    prompt: str,
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
) -> tuple[float, float]:
    """
    Return ``(quality_prob, adherence)`` in [0, 1]. On failure returns ``(0.5, 0.5)``.
    """
    try:
        import torch
        from vit_quality.checkpoint_utils import cached_vit_scorer

        scorer = cached_vit_scorer(vit_ckpt, device)
        with Image.open(image_path) as pil:
            out = scorer(scorer.image_tensor(pil).unsqueeze(0), prompt)
        q = float(torch.sigmoid(out["quality_logit"]).item())
        a = float(out["adherence_score"].item())
        if use_adherence:
            return q, float(0.65 * q + 0.35 * a)
        return q, a
    except Exception:
        return 0.5, 0.5


def blended_reward(
    composite: float,
    vit_score: float,
    *,
    vit_weight: float = 0.45,
) -> float:
    cw = max(0.0, 1.0 - vit_weight)
    return float(cw * composite + vit_weight * vit_score)


def enrich_results_with_vit(
    rows: Sequence[dict[str, Any]],
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
    vit_weight: float = 0.45,
) -> list[dict[str, Any]]:
    """Add ``vit_quality``, ``vit_reward``, ``blended_score`` to each result row."""
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        path = str(row.get("output", "") or "")
        prompt = str(row.get("prompt", "") or "")
        comp = float(row.get("composite", 0.0) or 0.0)
        if path and Path(path).is_file():
            _q, vit_r = score_image_vit(path, prompt, vit_ckpt=vit_ckpt, device=device, use_adherence=use_adherence)
            row["vit_quality"] = _q
            row["vit_reward"] = vit_r
        else:
            row["vit_quality"] = 0.5
            row["vit_reward"] = 0.5
        row["blended_score"] = blended_reward(comp, float(row["vit_reward"]), vit_weight=vit_weight)
        out.append(row)
    return out


def mine_vit_preference_pairs(
    rows: Sequence[dict[str, Any]],
    cfg: ViTMineConfig,
) -> list[dict[str, Any]]:
    """Mine win/lose pairs ranked by ``blended_score`` (ViT + composite)."""
    enriched = enrich_results_with_vit(
        rows,
        vit_ckpt=cfg.vit_ckpt,
        device=cfg.device,
        use_adherence=cfg.use_adherence,
        vit_weight=cfg.vit_weight,
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in enriched:
        case = str(r.get("case", "") or "")
        prompt = str(r.get("prompt", "") or "")
        if not prompt:
            continue
        groups.setdefault((case, prompt), []).append(r)

    pairs: list[dict[str, Any]] = []
    for (case, prompt), items in groups.items():
        scored = sorted(items, key=lambda x: float(x.get("blended_score", 0.0)), reverse=True)
        if len(scored) < 2:
            continue
        hi, lo_best = scored[0], scored[-1]
        margin = float(hi.get("blended_score", 0)) - float(lo_best.get("blended_score", 0))
        if margin < cfg.min_margin:
            continue
        win_p = str(hi.get("output", "") or "")
        lose_p = str(lo_best.get("output", "") or "")
        if not win_p or not lose_p:
            continue
        pairs.append(
            {
                "win_image_path": win_p,
                "lose_image_path": lose_p,
                "caption": prompt,
                "case": case,
                "win_score": float(hi.get("blended_score", 0)),
                "lose_score": float(lo_best.get("blended_score", 0)),
                "margin": margin,
                "source": "vit_mining",
                "vit_weight": cfg.vit_weight,
            }
        )
        used = 1
        for lo in reversed(scored[1:-1]):
            if used >= cfg.max_pairs_per_case:
                break
            m = float(hi.get("blended_score", 0)) - float(lo.get("blended_score", 0))
            if m < cfg.min_margin:
                continue
            lp = str(lo.get("output", "") or "")
            if not lp:
                continue
            pairs.append(
                {
                    "win_image_path": win_p,
                    "lose_image_path": lp,
                    "caption": prompt,
                    "case": case,
                    "win_score": float(hi.get("blended_score", 0)),
                    "lose_score": float(lo.get("blended_score", 0)),
                    "margin": m,
                    "source": "vit_mining",
                    "vit_weight": cfg.vit_weight,
                }
            )
            used += 1
    return pairs


# ==================================================================================================
# model_soup: **Model soup**: average multiple DiT checkpoints for robust inference weights.
#
# Same architecture + compatible keys only; useful after multi-seed training or DPO + base blend.
# ==================================================================================================


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(sd, dict):
        raise ValueError(f"No model/ema state in {path}")
    return {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k, v in sd.items()}


def average_state_dicts(
    states: Sequence[dict[str, torch.Tensor]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Element-wise weighted average of compatible state dicts."""
    if not states:
        raise ValueError("states must be non-empty")
    keys = set(states[0].keys())
    for s in states[1:]:
        keys &= set(s.keys())
    if not keys:
        raise ValueError("No shared keys across checkpoints")
    w = list(weights) if weights is not None else [1.0 / len(states)] * len(states)
    if len(w) != len(states):
        raise ValueError("weights length must match states")
    ws = sum(w) or 1.0
    w = [x / ws for x in w]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for i, st in enumerate(states):
            t = st[k]
            if not t.is_floating_point():
                out[k] = t.clone()
                acc = None
                break
            part = t.float() * w[i]
            acc = part if acc is None else acc + part
        if acc is not None:
            out[k] = acc.to(dtype=states[0][k].dtype)
    return out


def soup_checkpoints(
    paths: Iterable[str | Path],
    *,
    weights: Sequence[float] | None = None,
    template_ckpt: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load and average multiple ``.pt`` checkpoints."""
    plist = [Path(p) for p in paths]
    states = [_load_state(p) for p in plist]
    return average_state_dicts(states, weights=weights)


def save_soup_checkpoint(
    averaged: dict[str, torch.Tensor],
    *,
    template_path: str | Path,
    out_path: str | Path,
    tag: str = "model_soup",
) -> Path:
    """Write soup weights using metadata from ``template_path``."""
    tpl = torch.load(str(template_path), map_location="cpu", weights_only=False)
    tpl["model"] = averaged
    tpl["ema"] = averaged
    meta = tpl.get("meta") if isinstance(tpl.get("meta"), dict) else {}
    meta["model_soup"] = tag
    tpl["meta"] = meta
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tpl, op)
    return op


# ==================================================================================================
# eval_report: Generate **Markdown eval reports** from ``benchmark_suite`` outputs.
# ==================================================================================================


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_scaffold_summary() -> list[str]:
    """Lines for eval report: local config scaffolds vs hub fallback."""
    try:
        from utils.modeling.hf_scaffold import has_local_weights, scaffold_registry
        from utils.modeling.model_paths import model_dir

        local_cfg: list[str] = []
        local_w: list[str] = []
        for e in scaffold_registry():
            fp = model_dir() / e.name
            if not fp.is_dir() or not any(fp.iterdir()):
                continue
            if has_local_weights(fp):
                local_w.append(e.name)
            else:
                local_cfg.append(e.name)
        lines: list[str] = []
        if local_w:
            suffix = " …" if len(local_w) > 12 else ""
            lines.append(f"- Weights present ({len(local_w)}): {', '.join(local_w[:12])}{suffix}")
        if local_cfg:
            suffix = " …" if len(local_cfg) > 12 else ""
            lines.append(f"- Config-only scaffolds ({len(local_cfg)}): {', '.join(local_cfg[:12])}{suffix}")
        return lines
    except Exception:
        return []


def build_markdown_report(bench_dir: str | Path) -> str:
    """Build a human-readable report from ``results.json`` + ``leaderboard.json``."""
    d = Path(bench_dir)
    results = _load_json(d / "results.json") or []
    leaderboard = _load_json(d / "leaderboard.json") or []

    lines: list[str] = [
        "# SDX Benchmark Report",
        "",
        f"Directory: `{d}`",
        "",
        "## Leaderboard",
        "",
        "| Model | Mean composite | Std | Robust score | Cases |",
        "|-------|----------------|-----|--------------|-------|",
    ]
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('model', '')} | {float(row.get('mean_composite', 0)):.3f} | "
            f"{float(row.get('std_composite', 0)):.3f} | {float(row.get('robust_score', 0)):.3f} | "
            f"{int(row.get('cases', 0))} |"
        )

    # Per-case worst scores
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        case = str(r.get("case", "") or "unknown")
        by_case.setdefault(case, []).append(r)

    lines.extend(["", "## Weakest cases (lowest composite)", ""])
    case_avgs: list[tuple[str, float]] = []
    for case, rows in by_case.items():
        vals = [float(x.get("composite", 0)) for x in rows]
        if vals:
            case_avgs.append((case, sum(vals) / len(vals)))
    case_avgs.sort(key=lambda x: x[1])
    for case, avg in case_avgs[:8]:
        lines.append(f"- **{case}**: avg composite {avg:.3f}")

    if results:
        lines.extend(["", "## Hard-negative hint", ""])
        try:
            bundle = load_hard_negatives_from_results(d / "results.json")
            if bundle.negative_suffix:
                lines.append(f"Suggested extra negatives: `{bundle.negative_suffix[:200]}`")
        except Exception:
            pass

    hf_lines = _hf_scaffold_summary()
    if hf_lines:
        lines.extend(["", "## HF pretrained status", ""])
        lines.extend(hf_lines)

    try:
        from utils.modeling.text_encoder_stack import stack_download_hint, stack_status, stack_status_lines

        st = stack_status("penta")
        lines.extend(["", "## Text encoder stack (penta)", ""])
        lines.extend(stack_status_lines("penta"))
        if st.weights_count < len(st.slots):
            lines.append(f"- Download: `{stack_download_hint('penta')}`")
    except Exception:
        pass

    if results:
        modes = {
            str(r.get("text_encoder_mode", "")) for r in results if isinstance(r, dict) and r.get("text_encoder_mode")
        }
        modes.discard("")
        if modes:
            lines.extend(["", "## Checkpoint text encoder modes (from benchmark)", ""])
            for m in sorted(modes):
                try:
                    from utils.modeling.text_encoder_stack import text_encoder_mode_label

                    lines.append(f"- **{m}**: {text_encoder_mode_label(m)}")
                except Exception:
                    lines.append(f"- **{m}**")

    try:
        from utils.modeling.hf_scaffold import registry_summary

        s = registry_summary()
        lines.extend(
            [
                "",
                "## HF registry summary",
                "",
                f"- Total registry entries: {s['total_registry']}",
                f"- Local folders: {s['local_folders']} (weights: {s['with_weights']}, config-only: {s['config_only']})",
            ]
        )
        role_counts = s.get("role_counts", {})
        if isinstance(role_counts, dict) and role_counts:
            top = ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items()))
            lines.append(f"- By role: {top}")
    except Exception:
        pass

    try:
        from utils.modeling.hf_control import controlnet_has_weights, list_controlnet_types

        ctrl_ready = [t for t in list_controlnet_types() if controlnet_has_weights(t)]
        if ctrl_ready:
            lines.extend(["", "## ControlNet weights available", ""])
            lines.append(f"- {', '.join(ctrl_ready)}")
    except Exception:
        pass

    try:
        from utils.modeling.hf_scaffold import FACE_RESTORE_MODELS, UPSCALE_MODELS, has_local_weights

        up = [n for n, path in UPSCALE_MODELS.items() if has_local_weights(path())]
        face = [n for n, path in FACE_RESTORE_MODELS.items() if has_local_weights(path())]
        if up or face:
            lines.extend(["", "## Upscale / face-restore weights", ""])
            if up:
                lines.append(f"- Upscale: {', '.join(up)}")
            if face:
                lines.append(f"- Face restore: {', '.join(face)}")
    except Exception:
        pass

    # Optional per-result critic / HF reward columns
    hf_keys = ("natural", "art", "hf_reward", "hpsv2", "pickscore", "clip_h14")
    if results and any(isinstance(r, dict) and any(k in r for k in hf_keys) for r in results):
        lines.extend(["", "## Critic and reward averages (when logged in results.json)", ""])
        for key in hf_keys:
            vals = [float(r[key]) for r in results if isinstance(r, dict) and r.get(key) is not None]
            if vals:
                lines.append(f"- **{key}**: mean {sum(vals) / len(vals):.3f} (n={len(vals)})")

    lines.append("")
    return "\n".join(lines)


def write_report(bench_dir: str | Path, out_path: str | Path) -> Path:
    md = build_markdown_report(bench_dir)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(md, encoding="utf-8")
    return op


__all__ = [
    "AutoImproveConfig",
    "build_auto_improve_argv",
    "run_auto_improve",
    "run_tool",
    "DPOStageConfig",
    "MinePairsConfig",
    "build_dpo_train_argv",
    "mine_pairs_from_benchmark",
    "run_alignment_loop",
    "run_dpo_training",
    "curate_manifest",
    "run_flywheel",
    "HardNegativeBundle",
    "benchmark_sample_args_for_negatives",
    "load_hard_negatives_from_results",
    "mine_hard_negatives",
    "tags_from_benchmark_row",
    "ViTMineConfig",
    "blended_reward",
    "enrich_results_with_vit",
    "mine_vit_preference_pairs",
    "score_image_vit",
    "average_state_dicts",
    "save_soup_checkpoint",
    "soup_checkpoints",
    "build_markdown_report",
    "write_report",
]
