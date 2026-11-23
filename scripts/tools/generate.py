#!/usr/bin/env python3
"""Generation wrappers around ``sample.py`` and side pipelines.

    python -m scripts.tools generate hybrid --ckpt best.pt --prompt "..." --out out.png --vit-ckpt vit.pt --iterations 3
    python -m scripts.tools generate orchestrate --ckpt best.pt --prompt "a neon sign reading HELLO" --num 4
    python -m scripts.tools generate auto-refine --candidates 3 -- --ckpt best.pt --prompt "..."
    python -m scripts.tools generate edit-inpaint --ckpt best.pt --prompt "..." --init-image in.png --out out.png
    python -m scripts.tools generate editing-phase --image out.png --prompt "a samurai at sunset" --dry-run
    python -m scripts.tools generate cascade --prompt "..." --out cascade_out.png
    python -m scripts.tools generate profile-image path/to/image.png --json

``hybrid`` is DiT sampling + optional ViT reranking, incl. Tri-Consensus Iterative Synthesis (TCIS): DiT(+AR)
proposes candidates, the ViT quality/adherence heads act as a committee critic, a disagreement-aware consensus
score picks elites, and optional prompt self-correction runs over several iterations. Unknown ``hybrid`` flags
pass through to ``sample.py``.

Old names (``hybrid_dit_vit_generate``, ``orchestrate_pipeline``, ``auto_refine``, ``edit_inpaint``,
``editing_phase``, ``cascade_generate``, ``profile_image_cli``) still work as aliases.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PROG = "python -m scripts.tools generate"


def _parser(cmd: str, description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=f"{_PROG} {cmd}", description=description)


def _candidate_paths(out_path: Path, num: int) -> list[Path]:
    stem, ext = out_path.stem, out_path.suffix
    return [out_path.parent / f"{stem}_cand{i}{ext}" for i in range(max(0, int(num)))]


def fuse_vit_scores(
    quality_prob: float,
    adherence_score: float,
    *,
    quality_weight: float,
    adherence_weight: float,
) -> float:
    qw = max(0.0, float(quality_weight))
    aw = max(0.0, float(adherence_weight))
    z = qw + aw
    if z <= 0:
        qw, aw = 0.5, 0.5
    else:
        qw, aw = qw / z, aw / z
    return float(qw * float(quality_prob) + aw * float(adherence_score))


def consensus_score(
    quality_prob: float,
    adherence_score: float,
    *,
    quality_weight: float,
    adherence_weight: float,
    disagreement_penalty: float,
) -> float:
    """
    Committee consensus score.

    We want high quality and high adherence, but we penalize disagreement
    between those two heads to prefer stable, reliable candidates.
    """
    fused = fuse_vit_scores(
        quality_prob,
        adherence_score,
        quality_weight=quality_weight,
        adherence_weight=adherence_weight,
    )
    d = abs(float(quality_prob) - float(adherence_score))
    return float(fused - max(0.0, float(disagreement_penalty)) * d)


def constraint_consensus_score(
    *,
    quality_prob: float,
    adherence_score: float,
    ocr_score: float,
    count_score: float,
    saturation_score: float,
    quality_weight: float,
    adherence_weight: float,
    disagreement_penalty: float,
    ocr_weight: float,
    count_weight: float,
    saturation_weight: float,
) -> float:
    base = consensus_score(
        quality_prob,
        adherence_score,
        quality_weight=quality_weight,
        adherence_weight=adherence_weight,
        disagreement_penalty=disagreement_penalty,
    )
    return float(
        base
        + max(0.0, float(ocr_weight)) * float(ocr_score)
        + max(0.0, float(count_weight)) * float(count_score)
        + max(0.0, float(saturation_weight)) * float(saturation_score)
    )


def extract_expected_text(prompt: str) -> str:
    p = str(prompt or "")
    m = re.search(r'"([^"]{2,80})"', p)
    if m:
        return str(m.group(1)).strip()
    m2 = re.search(r"'([^']{2,80})'", p)
    if m2:
        return str(m2.group(1)).strip()
    return ""


def pareto_front_rows(rows: Sequence[dict[str, float]], objective_keys: Sequence[str]) -> list[dict[str, float]]:
    """
    Return non-dominated rows for max objectives.
    """
    keys = [k for k in objective_keys if k]
    if not rows or not keys:
        return list(rows)
    front: list[dict[str, float]] = []
    for i, r in enumerate(rows):
        dominated = False
        for j, s in enumerate(rows):
            if i == j:
                continue
            ge_all = True
            gt_any = False
            for k in keys:
                rv = float(r.get(k, 0.0))
                sv = float(s.get(k, 0.0))
                if sv < rv:
                    ge_all = False
                    break
                if sv > rv:
                    gt_any = True
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            front.append(dict(r))
    return front


def next_candidate_budget(current_num: int, best_consensus: float, *, threshold: float, step: int, max_num: int) -> int:
    cur = max(1, int(current_num))
    mx = max(cur, int(max_num))
    if float(best_consensus) >= float(threshold):
        return cur
    return min(mx, cur + max(1, int(step)))


def anneal_weight(base_weight: float, iteration_idx: int, total_iterations: int, mode: str = "up") -> float:
    base = float(base_weight)
    mode_n = str(mode or "none").strip().lower()
    if mode_n in {"none", ""} or int(total_iterations) <= 1:
        return base
    t = float(max(0, int(iteration_idx))) / float(max(1, int(total_iterations) - 1))
    if mode_n == "up":
        return base * (0.5 + 0.5 * t)
    if mode_n == "down":
        return base * (1.0 - 0.5 * t)
    return base


def uncertainty_score(best_row: dict[str, float], second_row: dict[str, float] | None = None) -> float:
    """
    Estimate decision uncertainty in [0,1]; higher means "spend more compute".
    """
    conf = float(best_row.get("vit_consensus_score", 0.0))
    q = float(best_row.get("vit_quality_prob", 0.0))
    a = float(best_row.get("vit_adherence_score", 0.0))
    cmin = min(
        float(best_row.get("ocr_score", 0.5)),
        float(best_row.get("count_score", 0.5)),
        float(best_row.get("saturation_score", 0.5)),
    )
    gap = 0.0
    if second_row is not None:
        gap = max(0.0, conf - float(second_row.get("vit_consensus_score", 0.0)))
    gap_term = 1.0 - min(1.0, gap / 0.25)
    disagree = abs(q - a)
    u = 0.45 * (1.0 - max(0.0, min(1.0, conf))) + 0.25 * gap_term + 0.2 * disagree + 0.1 * (1.0 - cmin)
    return float(max(0.0, min(1.0, u)))


def _signature(path: str | Path):
    import numpy as np
    from utils.quality.critics import image_signature

    return image_signature(np.asarray(Image.open(path).convert("RGB")))


def signature_novelty(signature, memory: Sequence) -> float:
    """Nearest-memory ``variation_guard`` distance scaled to [0, 1]; farther means more novel."""
    from utils.quality.critics import signature_distance

    if not memory:
        return 1.0
    return float(min(1.0, 3.0 * min(signature_distance(signature, m) for m in memory)))


def seed_for_iteration(base_seed: int, iteration_idx: int, seed_stride: int) -> int:
    return int(base_seed + max(0, int(iteration_idx)) * max(1, int(seed_stride)))


def maybe_self_correct_prompt(
    prompt: str,
    *,
    best_adherence: float,
    threshold: float,
    enable: bool,
) -> str:
    if not enable:
        return str(prompt)
    if float(best_adherence) >= float(threshold):
        return str(prompt)
    # Keep this concise and generic so it works across domains.
    corrective = "precise prompt adherence, exact object relationships, coherent anatomy, clean composition"
    p = str(prompt).strip()
    if corrective in p:
        return p
    return f"{corrective}, {p}".strip()


def reflective_prompt_update(prompt: str, best_row: dict[str, float], *, enable: bool) -> str:
    if not enable:
        return str(prompt)
    p = str(prompt).strip()
    additions: list[str] = []
    if float(best_row.get("vit_adherence_score", 0.0)) < 0.58:
        additions.append("literal prompt adherence, exact attribute binding")
    if float(best_row.get("count_score", 0.5)) < 0.45:
        additions.append("exact object/person counts, no extras, no missing subjects")
    if float(best_row.get("ocr_score", 0.5)) < 0.45:
        additions.append("clear legible typography, correct spelling")
    if float(best_row.get("saturation_score", 0.5)) < 0.35:
        additions.append("natural saturation, controlled color grading, no neon clipping")
    if not additions:
        return p
    add = ", ".join(additions)
    if add in p:
        return p
    return f"{add}, {p}".strip()


@functools.lru_cache(maxsize=1)
def _vit_scorer(ckpt: str, device: str, use_ema: bool):
    """Load the ViT critic once per process (TCIS reuses it every iteration)."""
    from vit_quality.model import ViTScorer

    return ViTScorer(ckpt, device, use_ema=use_ema)


def _score_candidates_with_vit(
    candidate_paths: Sequence[Path],
    *,
    prompt: str,
    vit_ckpt: str,
    vit_device: str,
    vit_use_ema: bool,
    default_num_ar_blocks: int | None,
    vit_quality_weight: float,
    vit_adherence_weight: float,
    vit_disagreement_penalty: float,
    consensus_ocr_weight: float,
    consensus_count_weight: float,
    consensus_saturation_weight: float,
    expected_text: str,
    expected_people_count: int,
    expected_object_count: int,
    expected_object_hint: str,
) -> list[dict[str, float]]:
    import numpy as np
    import torch
    from utils.modeling.architecture import normalize_num_ar_blocks
    from utils.quality.critics import score_oversaturation
    from utils.quality.test_time_pick import (
        score_object_count_match,
        score_ocr_match,
        score_people_count_match,
    )

    scorer = _vit_scorer(str(vit_ckpt), str(vit_device), bool(vit_use_ema))
    num_ar_blocks = normalize_num_ar_blocks(default_num_ar_blocks)

    rows: list[dict[str, float]] = []
    for p in candidate_paths:
        if not p.is_file():
            continue
        img = Image.open(p).convert("RGB")
        out = scorer(scorer.image_tensor(img).unsqueeze(0), str(prompt), num_ar_blocks)
        q = float(torch.sigmoid(out["quality_logit"]).item())
        a = float(out["adherence_score"].item())
        rgb = np.array(img, dtype=np.uint8)
        ocr = float(score_ocr_match(rgb, expected_text)) if str(expected_text).strip() else 0.5
        people_s = float(score_people_count_match(rgb, int(expected_people_count or 0)))
        obj_s = float(score_object_count_match(rgb, int(expected_object_count or 0), str(expected_object_hint or "")))
        count_s = float(max(people_s, obj_s))
        sat = score_oversaturation(rgb)
        sat_s = 0.5 if sat is None else float(sat)
        fused = fuse_vit_scores(
            q,
            a,
            quality_weight=vit_quality_weight,
            adherence_weight=vit_adherence_weight,
        )
        rows.append(
            {
                "path": str(p),
                "vit_quality_prob": q,
                "vit_adherence_score": a,
                "vit_fused_score": fused,
                "ocr_score": ocr,
                "count_score": count_s,
                "saturation_score": sat_s,
                "vit_consensus_score": constraint_consensus_score(
                    quality_prob=q,
                    adherence_score=a,
                    ocr_score=ocr,
                    count_score=count_s,
                    saturation_score=sat_s,
                    quality_weight=vit_quality_weight,
                    adherence_weight=vit_adherence_weight,
                    disagreement_penalty=vit_disagreement_penalty,
                    ocr_weight=consensus_ocr_weight,
                    count_weight=consensus_count_weight,
                    saturation_weight=consensus_saturation_weight,
                ),
            }
        )
    rows.sort(key=lambda r: float(r["vit_consensus_score"]), reverse=True)
    return rows


def _run_sample_once(
    *,
    ckpt: str,
    prompt: str,
    out_path: Path,
    num: int,
    steps: int | None,
    width: int | None,
    height: int | None,
    pick_best: str,
    seed: int | None,
    passthrough_args: Sequence[str],
    extra_positive_prefix: str = "",
    extra_negative_add: str = "",
) -> int:
    sample_py = ROOT / "sample.py"
    prompt_eff = str(prompt).strip()
    if str(extra_positive_prefix).strip():
        prompt_eff = f"{str(extra_positive_prefix).strip()}, {prompt_eff}".strip()
    cmd = [
        sys.executable,
        str(sample_py),
        "--ckpt",
        str(ckpt),
        "--prompt",
        str(prompt_eff),
        "--out",
        str(out_path),
        "--num",
        str(max(1, int(num))),
    ]
    if steps is not None and int(steps) > 0:
        cmd += ["--steps", str(int(steps))]
    if width is not None and int(width) > 0:
        cmd += ["--width", str(int(width))]
    if height is not None and int(height) > 0:
        cmd += ["--height", str(int(height))]
    if str(pick_best or "none").strip().lower() != "none":
        cmd += ["--pick-best", str(pick_best).strip().lower()]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if int(num) > 1:
        cmd += ["--pick-save-all"]
    pt_args = list(passthrough_args)
    if str(extra_negative_add).strip():
        merged = False
        for i, tok in enumerate(pt_args[:-1]):
            if tok == "--negative-prompt":
                base = str(pt_args[i + 1] or "").strip()
                pt_args[i + 1] = f"{base}, {str(extra_negative_add).strip()}".strip(", ")
                merged = True
                break
        if not merged:
            pt_args += ["--negative-prompt", str(extra_negative_add).strip()]
    if pt_args:
        cmd += pt_args
    print("[hybrid_dit_vit] Running sample:", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=ROOT).returncode)


def _hybrid_main(argv: list[str]) -> int:
    p = _parser("hybrid", "Hybrid DiT+ViT generation loop (TCIS); unknown flags pass through to sample.py.")
    p.add_argument("--ckpt", required=True, help="DiT checkpoint path for sample.py")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=4, help="Number of candidates.")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--pick-best",
        type=str,
        default="combo_hq",
        help="sample.py candidate selector before optional ViT rerank (none|clip|edge|ocr|combo|combo_exposure|combo_structural|combo_hq|combo_count).",
    )
    p.add_argument("--vit-ckpt", type=str, default="", help="Optional ViT checkpoint for reranking.")
    p.add_argument("--vit-device", type=str, default="cuda")
    p.add_argument("--vit-use-ema", action="store_true")
    p.add_argument("--vit-quality-weight", type=float, default=0.55)
    p.add_argument("--vit-adherence-weight", type=float, default=0.45)
    p.add_argument(
        "--vit-disagreement-penalty",
        type=float,
        default=0.10,
        help="Penalty on |quality-adherence| in committee consensus score.",
    )
    p.add_argument("--iterations", type=int, default=1, help="TCIS iterations (>=1).")
    p.add_argument("--seed-start", type=int, default=1337)
    p.add_argument("--seed-stride", type=int, default=101)
    p.add_argument(
        "--self-correct-prompt",
        action="store_true",
        help="Enable prompt self-correction when adherence falls below threshold.",
    )
    p.add_argument("--self-correct-threshold", type=float, default=0.63)
    p.add_argument(
        "--auto-shape-scaffold",
        action="store_true",
        help="Infer a structured shape/composition scaffold from prompt each iteration.",
    )
    p.add_argument("--shape-scaffold-strength", type=float, default=1.0)
    p.add_argument("--shape-max-actors", type=int, default=4)
    p.add_argument(
        "--pareto-elite", action="store_true", help="Select iteration winner from Pareto front of objectives."
    )
    p.add_argument("--pareto-topk", type=int, default=4, help="Cap Pareto elite pool to top-K by consensus.")
    p.add_argument("--adaptive-num", action="store_true", help="Increase candidate count when consensus is weak.")
    p.add_argument("--adaptive-threshold", type=float, default=0.80)
    p.add_argument("--adaptive-step", type=int, default=2)
    p.add_argument("--adaptive-max-num", type=int, default=12)
    p.add_argument("--reflection-update", action="store_true", help="Use metric-aware prompt reflection updates.")
    p.add_argument("--consensus-ocr-weight", type=float, default=0.08)
    p.add_argument("--consensus-count-weight", type=float, default=0.08)
    p.add_argument("--consensus-saturation-weight", type=float, default=0.05)
    p.add_argument("--constraint-anneal", choices=("none", "up", "down"), default="up")
    p.add_argument(
        "--elite-memory-size", type=int, default=8, help="Cross-iteration elite memory size for diversity bonus."
    )
    p.add_argument("--diversity-bonus-weight", type=float, default=0.05)
    p.add_argument("--uncertainty-threshold", type=float, default=0.42)
    p.add_argument("--uncertainty-extra-iterations", type=int, default=1)
    p.add_argument("--uncertainty-max-iterations", type=int, default=8)
    p.add_argument("--expected-text", type=str, default="", help="Override expected text for OCR scoring.")
    p.add_argument("--expected-people-count", type=int, default=0, help="Override expected people count.")
    p.add_argument("--expected-object-count", type=int, default=0, help="Override expected object count.")
    p.add_argument("--expected-object-hint", type=str, default="", help="Object hint for object count scoring.")
    args, unknown = p.parse_known_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = max(1, int(args.iterations))
    iterations_runtime = iterations
    prompt_cur = str(args.prompt)

    from utils.modeling.architecture import read_num_ar_blocks_from_checkpoint

    default_ar = read_num_ar_blocks_from_checkpoint(args.ckpt)
    if default_ar < 0:
        default_ar = None

    iter_reports: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    cur_num = max(1, int(args.num))
    elite_memory: list[tuple[float, float, float, float, float, float]] = []
    i = 0
    while i < iterations_runtime:
        seed_i = seed_for_iteration(int(args.seed_start), i, int(args.seed_stride))
        iter_out = out_path.parent / f"{out_path.stem}_iter{i}{out_path.suffix}"
        shape_pos = ""
        shape_neg = ""
        shape_blueprint: dict[str, object] | None = None
        if bool(args.auto_shape_scaffold):
            from utils.prompt.scene import compile_shape_scaffold

            shape_pos, shape_neg, shape_blueprint = compile_shape_scaffold(
                prompt_cur,
                strength=float(args.shape_scaffold_strength),
                max_actors=int(args.shape_max_actors),
            )
        rc = _run_sample_once(
            ckpt=str(args.ckpt),
            prompt=prompt_cur,
            out_path=iter_out,
            num=int(cur_num),
            steps=args.steps,
            width=args.width,
            height=args.height,
            pick_best=str(args.pick_best),
            seed=seed_i,
            passthrough_args=unknown,
            extra_positive_prefix=shape_pos,
            extra_negative_add=shape_neg,
        )
        if rc != 0:
            return rc

        # If ViT not configured, keep last iteration output and return.
        if not str(args.vit_ckpt).strip() or int(args.num) <= 1:
            Image.open(iter_out).convert("RGB").save(out_path)
            iter_reports.append(
                {
                    "iteration": i,
                    "seed": seed_i,
                    "prompt": prompt_cur,
                    "vit_enabled": False,
                    "selected_path": str(iter_out),
                    "auto_shape_scaffold": bool(args.auto_shape_scaffold),
                    "shape_blueprint": shape_blueprint,
                }
            )
            i += 1
            continue

        cand = [p for p in _candidate_paths(iter_out, int(cur_num)) if p.is_file()]
        if not cand:
            print("[hybrid_dit_vit] No candidate files found for ViT rerank in this iteration.")
            Image.open(iter_out).convert("RGB").save(out_path)
            iter_reports.append(
                {
                    "iteration": i,
                    "seed": seed_i,
                    "prompt": prompt_cur,
                    "vit_enabled": True,
                    "selected_path": str(iter_out),
                    "note": "no_candidates_for_vit",
                    "auto_shape_scaffold": bool(args.auto_shape_scaffold),
                    "shape_blueprint": shape_blueprint,
                }
            )
            i += 1
            continue

        exp_text = str(args.expected_text or "").strip() or extract_expected_text(prompt_cur)
        if int(args.expected_people_count) > 0:
            exp_people = int(args.expected_people_count)
        else:
            from utils.quality.test_time_pick import infer_expected_people_count

            exp_people = int(infer_expected_people_count(prompt_cur))
        if int(args.expected_object_count) > 0:
            exp_obj = int(args.expected_object_count)
            exp_obj_hint = str(args.expected_object_hint or "").strip()
        else:
            from utils.quality.test_time_pick import infer_expected_object_count

            exp_obj, auto_hint = infer_expected_object_count(prompt_cur)
            exp_obj_hint = str(args.expected_object_hint or "").strip() or str(auto_hint or "")

        rows = _score_candidates_with_vit(
            cand,
            prompt=prompt_cur,
            vit_ckpt=str(args.vit_ckpt),
            vit_device=str(args.vit_device),
            vit_use_ema=bool(args.vit_use_ema),
            default_num_ar_blocks=default_ar,
            vit_quality_weight=float(args.vit_quality_weight),
            vit_adherence_weight=float(args.vit_adherence_weight),
            vit_disagreement_penalty=float(args.vit_disagreement_penalty),
            consensus_ocr_weight=anneal_weight(
                float(args.consensus_ocr_weight),
                i,
                iterations_runtime,
                mode=str(args.constraint_anneal),
            ),
            consensus_count_weight=anneal_weight(
                float(args.consensus_count_weight),
                i,
                iterations_runtime,
                mode=str(args.constraint_anneal),
            ),
            consensus_saturation_weight=anneal_weight(
                float(args.consensus_saturation_weight),
                i,
                iterations_runtime,
                mode=str(args.constraint_anneal),
            ),
            expected_text=exp_text,
            expected_people_count=exp_people,
            expected_object_count=int(exp_obj),
            expected_object_hint=exp_obj_hint,
        )
        if not rows:
            print("[hybrid_dit_vit] ViT produced no scores in this iteration; keeping DiT output.")
            Image.open(iter_out).convert("RGB").save(out_path)
            i += 1
            continue

        selection_pool = list(rows)
        if bool(args.pareto_elite):
            objective_keys = [
                "vit_quality_prob",
                "vit_adherence_score",
                "ocr_score",
                "count_score",
                "saturation_score",
            ]
            pareto = pareto_front_rows(rows, objective_keys)
            pareto.sort(key=lambda r: float(r.get("vit_consensus_score", 0.0)), reverse=True)
            k = max(1, int(args.pareto_topk))
            selection_pool = pareto[:k] if pareto else selection_pool

        # V4: Cross-iteration elite memory adds diversity pressure.
        selection_pool_enriched: list[dict[str, float]] = []
        sigs: dict[str, object] = {}
        for r in selection_pool:
            rr = dict(r)
            try:
                sigs[str(rr["path"])] = _signature(str(rr["path"]))
                bonus = signature_novelty(sigs[str(rr["path"])], elite_memory)
            except Exception:
                bonus = 0.0
            rr["v4_diversity_bonus"] = float(bonus)
            rr["v4_final_score"] = float(rr.get("vit_consensus_score", 0.0)) + float(
                args.diversity_bonus_weight
            ) * float(bonus)
            selection_pool_enriched.append(rr)
        selection_pool_enriched.sort(key=lambda r: float(r.get("v4_final_score", 0.0)), reverse=True)
        selection_pool = selection_pool_enriched or selection_pool

        best = dict(selection_pool[0])
        best_path = Path(str(best["path"]))
        if best_path.is_file():
            Image.open(best_path).convert("RGB").save(out_path)
            try:
                sig = sigs.get(str(best_path))
                elite_memory.append(_signature(best_path) if sig is None else sig)
                k_mem = max(1, int(args.elite_memory_size))
                if len(elite_memory) > k_mem:
                    elite_memory = elite_memory[-k_mem:]
            except Exception:
                pass

        prompt_next = maybe_self_correct_prompt(
            prompt_cur,
            best_adherence=float(best.get("vit_adherence_score", 0.0)),
            threshold=float(args.self_correct_threshold),
            enable=bool(args.self_correct_prompt),
        )
        prompt_next = reflective_prompt_update(prompt_next, best, enable=bool(args.reflection_update))
        iter_report = {
            "iteration": i,
            "seed": seed_i,
            "prompt": prompt_cur,
            "selected_path": str(best_path),
            "vit_quality_prob": float(best.get("vit_quality_prob", 0.0)),
            "vit_adherence_score": float(best.get("vit_adherence_score", 0.0)),
            "vit_fused_score": float(best.get("vit_fused_score", 0.0)),
            "vit_consensus_score": float(best.get("vit_consensus_score", 0.0)),
            "v4_final_score": float(best.get("v4_final_score", best.get("vit_consensus_score", 0.0))),
            "v4_diversity_bonus": float(best.get("v4_diversity_bonus", 0.0)),
            "ocr_score": float(best.get("ocr_score", 0.5)),
            "count_score": float(best.get("count_score", 0.5)),
            "saturation_score": float(best.get("saturation_score", 0.5)),
            "expected_text": exp_text,
            "expected_people_count": exp_people,
            "expected_object_count": int(exp_obj),
            "expected_object_hint": exp_obj_hint,
            "self_corrected_prompt": prompt_next if prompt_next != prompt_cur else None,
            "auto_shape_scaffold": bool(args.auto_shape_scaffold),
            "shape_blueprint": shape_blueprint,
            "ranking": rows,
            "pareto_elite_enabled": bool(args.pareto_elite),
            "pareto_pool_size": len(selection_pool),
        }
        iter_reports.append(iter_report)
        row_with_iter = dict(best)
        row_with_iter["iteration"] = i
        row_with_iter["seed"] = seed_i
        row_with_iter["v4_final_score"] = float(best.get("v4_final_score", best.get("vit_consensus_score", 0.0)))
        global_rows.append(row_with_iter)
        prompt_cur = prompt_next
        second_row = selection_pool[1] if len(selection_pool) > 1 else None
        u = uncertainty_score(best, second_row)
        iter_report["v4_uncertainty"] = float(u)
        if bool(args.adaptive_num):
            cur_num = next_candidate_budget(
                cur_num,
                float(best.get("vit_consensus_score", 0.0)),
                threshold=float(args.adaptive_threshold),
                step=int(args.adaptive_step),
                max_num=int(args.adaptive_max_num),
            )
        if (
            float(u) >= float(args.uncertainty_threshold)
            and iterations_runtime < int(args.uncertainty_max_iterations)
            and int(args.uncertainty_extra_iterations) > 0
        ):
            iterations_runtime = min(
                int(args.uncertainty_max_iterations),
                iterations_runtime + int(args.uncertainty_extra_iterations),
            )
        i += 1

    if global_rows:
        global_rows.sort(key=lambda r: float(r.get("v4_final_score", r.get("vit_consensus_score", 0.0))), reverse=True)
        best_global = global_rows[0]
        best_global_path = Path(str(best_global["path"]))
        if best_global_path.is_file():
            Image.open(best_global_path).convert("RGB").save(out_path)
        print(
            "[hybrid_dit_vit] TCIS selected",
            best_global_path.name,
            f"(consensus={float(best_global.get('vit_consensus_score', 0.0)):.4f}) -> {out_path}",
        )

    sidecar = out_path.parent / f"{out_path.stem}_hybrid_scores.json"
    sidecar.write_text(
        json.dumps(
            {
                "mode": "tcis",
                "iterations_requested": iterations,
                "iterations_effective": iterations_runtime,
                "vit_enabled": bool(str(args.vit_ckpt).strip()),
                "iter_reports": iter_reports,
                "global_best": global_rows[0] if global_rows else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[hybrid_dit_vit] Scores: {sidecar}")
    return 0


def _orchestrate_main(argv: list[str]) -> int:
    """Designer -> Verifier (LANDSCAPE §2): K candidates via sample.py ``--pick-best``, save the winner."""
    p = _parser("orchestrate", "Run sample.py with K candidates and pick-best.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative-prompt", type=str, default="", dest="negative_prompt")
    p.add_argument("--out", type=str, default="orch_out.png")
    p.add_argument("--num", type=int, default=4, help="Number of candidates (>=2 for pick-best)")
    p.add_argument(
        "--pick-best",
        type=str,
        default="combo",
        choices=["none", "clip", "edge", "ocr", "combo", "combo_exposure"],
        dest="pick_best",
    )
    p.add_argument("--expected-text", type=str, default="", help="For ocr / combo when text-in-image")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--pick-save-all", action="store_true", help="Save all candidates as stem_cand{i}")
    p.add_argument("--no-refine", action="store_true", help="Pass --no-refine to sample.py")
    p.add_argument("--extra", nargs="*", default=[], help="Extra args passed through to sample.py")
    args = p.parse_args(argv)

    cmd = [
        sys.executable,
        str(ROOT / "sample.py"),
        "--ckpt",
        args.ckpt,
        "--prompt",
        args.prompt,
        "--out",
        args.out,
        "--num",
        str(max(1, args.num)),
        "--pick-best",
        args.pick_best,
        "--steps",
        str(args.steps),
        "--cfg-scale",
        str(args.cfg_scale),
    ]
    if args.negative_prompt:
        cmd += ["--negative-prompt", args.negative_prompt]
    if args.seed >= 0:
        cmd += ["--seed", str(args.seed)]
    if args.expected_text:
        cmd += ["--expected-text", args.expected_text]
    if args.pick_save_all:
        cmd.append("--pick-save-all")
    if args.no_refine:
        cmd.append("--no-refine")
    cmd.extend(args.extra)
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _sample_passthrough(raw: list[str]) -> list[str]:
    return raw[1:] if raw and raw[0] == "--" else list(raw)


def _auto_refine_main(argv: list[str]) -> int:
    from utils.generation.sample_prompt import run_auto_refine_candidates

    p = _parser("auto-refine", "Run sample.py N times with seed offsets; keep best output by heuristic score.")
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--out", type=str, default="refined.png")
    p.add_argument("--stem", type=str, default="refine_candidate")
    p.add_argument("sample_args", nargs=argparse.REMAINDER, help="Args passed to sample.py (after --)")
    args = p.parse_args(argv)
    res = run_auto_refine_candidates(
        sample_cmd=[sys.executable, str(ROOT / "sample.py"), *_sample_passthrough(args.sample_args)],
        num_candidates=args.candidates,
        seed_base=args.seed_base,
        out_stem=args.stem,
    )
    shutil.copy(res.outputs[res.best_index], args.out)
    print(f"Best candidate {res.best_index} score={res.scores[res.best_index]:.3f} -> {args.out}")
    return 0


_HEURISTIC_CHOICES = ("face", "hands", "clothing", "background", "subject", "full")


def _positive_int(label: str, minimum: int = 1):
    def _checker(s: str) -> int:
        v = int(s, 10)
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{label} must be >= {minimum}, got {v}")
        return v

    return _checker


def _unit_interval(s: str) -> float:
    v = float(s)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"value must be in [0, 1], got {v}")
    return v


def _edit_inpaint_main(argv: list[str]) -> int:
    ap = _parser(
        "edit-inpaint", "SDX: img2img or inpaint via sample.py (--init-image; optional mask or heuristic region)."
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output image path")
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--init-image", type=str, required=True, help="Base RGB image path")
    ap.add_argument(
        "--mask", type=str, default="", help="Grayscale mask (white = regenerate). Omit for whole-image img2img."
    )
    ap.add_argument(
        "--heuristic-mask-region",
        type=str,
        default="",
        choices=["", *_HEURISTIC_CHOICES],
        metavar="REGION",
        help="Coarse inpaint mask (see utils/generation/editing.py). Alternative to --mask.",
    )
    ap.add_argument("--width", type=_positive_int("width"), default=512)
    ap.add_argument("--height", type=_positive_int("height"), default=512)
    ap.add_argument("--steps", type=_positive_int("steps"), default=28)
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--strength", type=_unit_interval, default=0.65, help="Img2img / inpaint strength in [0, 1]")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--scheduler", type=str, default="ddim")
    ap.add_argument("--solver", type=str, default="ddim")
    ap.add_argument(
        "--inpaint-mode",
        type=str,
        default="mdm",
        choices=("legacy", "mdm"),
        help="Used only when a mask is supplied",
    )
    ap.add_argument(
        "--segment-for",
        type=str,
        default="",
        help=(
            "Infer mask from phrase (Grounding DINO + SAM2 under pretrained/ when available; "
            "else keywords). Exclusive with --mask and --heuristic-mask-region."
        ),
    )
    ap.add_argument(
        "--no-segment-models",
        action="store_true",
        help="For --segment-for: heuristic phrase mapping only; skip DINO/SAM pipelines.",
    )
    ap.add_argument("--from-pillow", action="store_true", help="Use PIL round-trip temp files (same as API runner)")
    args = ap.parse_args(argv)

    from utils.generation.editing import build_segmentation_mask_for_edit, heuristic_inpaint_mask
    from utils.generation.sample_argv import run_edit_with_pillow, run_sample_inference

    seed = None if args.seed < 0 else args.seed
    user_mask = str(args.mask or "").strip()
    heuristic = str(args.heuristic_mask_region or "").strip()
    segment_for = str(args.segment_for or "").strip()
    if sum(1 for x in (user_mask, heuristic, segment_for) if x) > 1:
        ap.error("Use at most one of --mask, --heuristic-mask-region, --segment-for.")

    base = Image.open(args.init_image).convert("RGB")
    # Build the mask once (segmentation models are expensive), then hand it over as PIL or as a temp PNG.
    mask_img: Image.Image | None = None
    if heuristic:
        mask_img = heuristic_inpaint_mask(args.width, args.height, heuristic)
    elif segment_for:
        guide = base.resize((args.width, args.height), Image.Resampling.LANCZOS)
        mask_img = build_segmentation_mask_for_edit(
            guide, segment_for, feather_radius=4.0, use_vision_models=not args.no_segment_models
        ).mask

    common = {
        "ckpt": args.ckpt,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": seed,
        "inpaint_mode": args.inpaint_mode,
        "device": args.device,
        "scheduler": args.scheduler,
        "solver": args.solver,
    }
    if args.from_pillow:
        if mask_img is None and user_mask:
            mask_img = Image.open(user_mask).convert("L")
        run_edit_with_pillow(base_image=base, mask_image=mask_img, img2img_strength=args.strength, **common).save(
            args.out
        )
        print(f"Saved: {args.out}", flush=True)
        return 0

    with tempfile.TemporaryDirectory(prefix="sdx_inpaint_mask_") as tmp:
        mask_path: str | None = user_mask or None
        if mask_img is not None:
            mask_path = str(Path(tmp) / "mask.png")
            mask_img.save(mask_path, format="PNG")
        run_sample_inference(
            out_path=args.out,
            init_image_path=args.init_image,
            mask_image_path=mask_path,
            strength=args.strength,
            **common,
        )
    print(f"Saved: {args.out}", flush=True)
    return 0


def _editing_phase_main(argv: list[str]) -> int:
    p = _parser("editing-phase", "SDX editing phase (diagnose -> pieces -> edit -> gate) until natural / cohesive.")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompt", required=True, help="Positive prompt used for generation")
    p.add_argument("--negative-prompt", default="", help="Negative prompt")
    p.add_argument("--ckpt", default="", help="Checkpoint for img2img/inpaint (optional with --dry-run)")
    p.add_argument("--out-dir", default="outputs/editing_phase", help="Work / output directory")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--min-clip", type=float, default=0.28)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Plan + masks + art_post only; no sample.py")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--no-art-post", action="store_true")
    p.add_argument("--caption", default="", help="Optional image caption for missing-token detection")
    p.add_argument("--scheduler", default="ays_dit")
    p.add_argument("--solver", default="dpmpp_2m")
    p.add_argument("--refine-steps", type=int, default=20)
    args = p.parse_args(argv)

    from utils.generation.editing import EditingPhaseConfig, run_editing_phase

    cfg = EditingPhaseConfig(
        max_iters=int(args.max_iters),
        min_clip=float(args.min_clip),
        enable_ocr=not bool(args.no_ocr),
        enable_art_post=not bool(args.no_art_post),
        dry_run=bool(args.dry_run) or not str(args.ckpt).strip(),
        device=str(args.device),
        scheduler=str(args.scheduler),
        solver=str(args.solver),
        refine_steps=int(args.refine_steps),
    )
    result = run_editing_phase(
        args.image,
        args.prompt,
        ckpt=str(args.ckpt).strip() or None,
        negative_prompt=str(args.negative_prompt),
        config=cfg,
        work_dir=args.out_dir,
        seed=args.seed,
        caption=str(args.caption or ""),
    )
    report = {
        "stopped_reason": result.stopped_reason,
        "iterations": result.iterations,
        "output_path": result.output_path,
        "piece_dir": result.piece_dir,
        "prompt": result.prompt,
        "actions": [
            {"kind": a.kind, "reason": a.reason, "region": a.region, "priority": a.priority}
            for a in result.actions_applied
        ],
        "diagnosis": [
            {
                "clip": d.clip_score,
                "sharpness": d.sharpness,
                "gate_passed": d.gate_passed,
                "failures": d.gate_failures,
                "missing_tokens": d.missing_tokens,
                "expected_text": d.expected_text,
                "pieces": d.piece_labels,
            }
            for d in result.diagnosis_history
        ],
    }
    report_path = Path(args.out_dir) / "editing_phase_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report -> {report_path}", file=sys.stderr)
    return 0


def _cascade_main(argv: list[str]) -> int:
    """Stable Cascade (diffusers) — optional path separate from DiT+T5 (models under model/StableCascade-*)."""
    from utils.modeling.model_paths import default_cascade_decoder_path, default_cascade_prior_path

    p = _parser("cascade", "Stable Cascade image generation (Diffusers)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="cascade_out.png")
    p.add_argument("--prior", type=str, default="", help="Prior model path (empty = default)")
    p.add_argument("--decoder", type=str, default="", help="Decoder model path (empty = default)")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args(argv)

    from utils.safety.minor_guard import is_sexual, require_allowed, require_image_allowed

    require_allowed(args.prompt)

    try:
        import torch
        from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    except ImportError:
        print("Install diffusers and torch: pip install diffusers", file=sys.stderr)
        return 1

    prior_id = args.prior or default_cascade_prior_path()
    dec_id = args.decoder or default_cascade_decoder_path()
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading prior: {prior_id}")
    prior = StableCascadePriorPipeline.from_pretrained(prior_id, torch_dtype=dtype).to(device)
    print(f"Loading decoder: {dec_id}")
    decoder = StableCascadeDecoderPipeline.from_pretrained(dec_id, torch_dtype=dtype).to(device)

    prompt = args.prompt
    prior_output = prior(prompt=prompt, height=1024, width=1024, guidance_scale=4.0, num_inference_steps=20)
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=0.0,
        output_type="pil",
    ).images[0]
    require_image_allowed(decoder_output, sexual=is_sexual(prompt))
    decoder_output.save(args.out)
    print(f"Saved: {args.out}")
    return 0


def _profile_image_main(argv: list[str]) -> int:
    from utils.captioning import profile_image

    p = _parser("profile-image", "Profile one image (booru + SauceNAO + VLM) to test the caption fusion pipeline.")
    p.add_argument("image", help="Image path")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--no-reverse", action="store_true")
    p.add_argument("--no-saucenao", action="store_true")
    p.add_argument("--no-tineye", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    prof = profile_image(
        args.image,
        use_vlm=not args.no_vlm,
        use_reverse_search=not args.no_reverse,
        use_saucenao=not args.no_saucenao,
        use_tineye=not args.no_tineye,
        device=args.device,
    )
    if args.json:
        print(json.dumps(prof.to_manifest_patch() | {"sources": prof.sources}, ensure_ascii=False, indent=2))
        return 0
    print("caption:", prof.caption)
    print("scene_summary:", prof.scene_summary)
    print("character:", prof.character_tags)
    print("copyright:", prof.copyright_tags)
    print("artist:", prof.artist_tags)
    print("is_oc:", prof.is_original_character)
    print("confidence:", prof.confidence)
    print("sources:", prof.sources)
    return 0


COMMANDS = {
    "hybrid": _hybrid_main,
    "orchestrate": _orchestrate_main,
    "auto-refine": _auto_refine_main,
    "edit-inpaint": _edit_inpaint_main,
    "editing-phase": _editing_phase_main,
    "cascade": _cascade_main,
    "profile-image": _profile_image_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
