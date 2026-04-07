#!/usr/bin/env python3
"""
Hybrid generator: DiT sampling + optional ViT reranking.

This script also supports a new design mode:
Tri-Consensus Iterative Synthesis (TCIS)
- DiT(+AR) proposes candidate sets,
- ViT quality/adherence acts as a committee critic,
- a disagreement-aware consensus score picks elites,
- optional prompt self-correction runs for multiple iterations.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _candidate_paths(out_path: Path, num: int) -> List[Path]:
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


def pareto_front_rows(rows: Sequence[Dict[str, float]], objective_keys: Sequence[str]) -> List[Dict[str, float]]:
    """
    Return non-dominated rows for max objectives.
    """
    keys = [k for k in objective_keys if k]
    if not rows or not keys:
        return list(rows)
    front: List[Dict[str, float]] = []
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


def uncertainty_score(best_row: Dict[str, float], second_row: Dict[str, float] | None = None) -> float:
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


def image_signature(path: str | Path) -> tuple[float, float, float, float, float, float]:
    """
    Tiny image descriptor for diversity memory: mean/std per RGB channel.
    """
    import numpy as np

    p = Path(path)
    img = Image.open(p).convert("RGB").resize((32, 32), resample=Image.BILINEAR)
    x = np.array(img, dtype=np.float32).reshape(-1, 3)
    m = x.mean(axis=0)
    s = x.std(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]), float(s[0]), float(s[1]), float(s[2]))


def signature_novelty(
    signature: tuple[float, float, float, float, float, float],
    memory: Sequence[tuple[float, float, float, float, float, float]],
) -> float:
    if not memory:
        return 1.0
    dists: List[float] = []
    for m in memory:
        dm = (
            abs(signature[0] - m[0])
            + abs(signature[1] - m[1])
            + abs(signature[2] - m[2])
            + abs(signature[3] - m[3])
            + abs(signature[4] - m[4])
            + abs(signature[5] - m[5])
        ) / (6.0 * 255.0)
        dists.append(float(dm))
    # Use nearest-memory distance; farther means more novel.
    nearest = min(dists) if dists else 0.0
    return float(max(0.0, min(1.0, nearest * 3.0)))


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


def reflective_prompt_update(prompt: str, best_row: Dict[str, float], *, enable: bool) -> str:
    if not enable:
        return str(prompt)
    p = str(prompt).strip()
    additions: List[str] = []
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
) -> List[Dict[str, float]]:
    import numpy as np
    from utils.architecture.ar_block_conditioning import ar_conditioning_vector, normalize_num_ar_blocks
    from utils.quality.test_time_pick import (
        score_object_count_match,
        score_ocr_match,
        score_people_count_match,
        score_saturation_balance,
    )
    from vit_quality.checkpoint_utils import load_vit_quality_checkpoint
    from vit_quality.dataset import text_feature_vector

    model, _cfg = load_vit_quality_checkpoint(vit_ckpt, use_ema=bool(vit_use_ema))
    device = torch.device(vit_device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    image_size = int(_cfg.get("image_size", 224))
    use_ar = bool(_cfg.get("use_ar_conditioning", False))
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    txt = text_feature_vector(str(prompt)).unsqueeze(0).to(device)

    ar_vec = None
    if use_ar:
        ar_b = -1
        if default_num_ar_blocks is not None:
            ar_b = normalize_num_ar_blocks(default_num_ar_blocks)
        ar_vec = ar_conditioning_vector(ar_b, device=device, dtype=txt.dtype).unsqueeze(0)

    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for p in candidate_paths:
            if not p.is_file():
                continue
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            out = model(x, txt, ar_vec)
            q = float(torch.sigmoid(out["quality_logit"]).item())
            a = float(out["adherence_score"].item())
            rgb = np.array(img, dtype=np.uint8)
            ocr = float(score_ocr_match(rgb, expected_text)) if str(expected_text).strip() else 0.5
            people_s = float(score_people_count_match(rgb, int(expected_people_count or 0)))
            obj_s = float(score_object_count_match(rgb, int(expected_object_count or 0), str(expected_object_hint or "")))
            count_s = float(max(people_s, obj_s))
            sat_s = float(score_saturation_balance(rgb))
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


def main() -> int:
    p = argparse.ArgumentParser(description="Hybrid DiT+ViT generation loop.")
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
    p.add_argument("--pareto-elite", action="store_true", help="Select iteration winner from Pareto front of objectives.")
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
    p.add_argument("--elite-memory-size", type=int, default=8, help="Cross-iteration elite memory size for diversity bonus.")
    p.add_argument("--diversity-bonus-weight", type=float, default=0.05)
    p.add_argument("--uncertainty-threshold", type=float, default=0.42)
    p.add_argument("--uncertainty-extra-iterations", type=int, default=1)
    p.add_argument("--uncertainty-max-iterations", type=int, default=8)
    p.add_argument("--expected-text", type=str, default="", help="Override expected text for OCR scoring.")
    p.add_argument("--expected-people-count", type=int, default=0, help="Override expected people count.")
    p.add_argument("--expected-object-count", type=int, default=0, help="Override expected object count.")
    p.add_argument("--expected-object-hint", type=str, default="", help="Object hint for object count scoring.")
    args, unknown = p.parse_known_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = max(1, int(args.iterations))
    iterations_runtime = iterations
    prompt_cur = str(args.prompt)

    from utils.architecture.ar_block_conditioning import read_num_ar_blocks_from_checkpoint

    default_ar = read_num_ar_blocks_from_checkpoint(args.ckpt)
    if default_ar < 0:
        default_ar = None

    iter_reports: List[Dict[str, object]] = []
    global_rows: List[Dict[str, object]] = []
    cur_num = max(1, int(args.num))
    elite_memory: List[tuple[float, float, float, float, float, float]] = []
    i = 0
    while i < iterations_runtime:
        seed_i = seed_for_iteration(int(args.seed_start), i, int(args.seed_stride))
        iter_out = out_path.parent / f"{out_path.stem}_iter{i}{out_path.suffix}"
        shape_pos = ""
        shape_neg = ""
        shape_blueprint: Dict[str, object] | None = None
        if bool(args.auto_shape_scaffold):
            from utils.prompt.shape_scaffold import compile_shape_scaffold

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
        selection_pool_enriched: List[Dict[str, float]] = []
        for r in selection_pool:
            rr = dict(r)
            bonus = 0.0
            try:
                sig = image_signature(str(rr["path"]))
                bonus = signature_novelty(sig, elite_memory)
            except Exception:
                bonus = 0.0
            rr["v4_diversity_bonus"] = float(bonus)
            rr["v4_final_score"] = float(rr.get("vit_consensus_score", 0.0)) + float(args.diversity_bonus_weight) * float(
                bonus
            )
            selection_pool_enriched.append(rr)
        selection_pool_enriched.sort(key=lambda r: float(r.get("v4_final_score", 0.0)), reverse=True)
        selection_pool = selection_pool_enriched or selection_pool

        best = dict(selection_pool[0])
        best_path = Path(str(best["path"]))
        if best_path.is_file():
            Image.open(best_path).convert("RGB").save(out_path)
            try:
                elite_memory.append(image_signature(best_path))
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


if __name__ == "__main__":
    raise SystemExit(main())
