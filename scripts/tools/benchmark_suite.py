"""
Benchmark multiple checkpoints on a structured prompt suite.

Usage:
    python -m scripts.tools.benchmark_suite --ckpt results/a/best.pt results/b/best.pt --preset sdxl
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from utils.quality import test_time_pick as ttp


@dataclass
class PromptCase:
    name: str
    prompt: str
    expected_text: str = ""
    expected_count: int = 0
    expected_count_target: str = "auto"  # auto|people|objects
    expected_count_object: str = ""
    width: int = 1024
    height: int = 1024


DEFAULT_SUITE: List[PromptCase] = [
    PromptCase(
        name="text_sign",
        prompt='street cafe storefront, neon sign that says "OPEN 24H", rainy night, cinematic',
        expected_text="OPEN 24H",
    ),
    PromptCase(
        name="people_count",
        prompt="exactly 3 people sitting at a table, medium shot, natural proportions",
        expected_count=3,
        expected_count_target="people",
    ),
    PromptCase(
        name="object_count",
        prompt="exactly 7 coins on a wooden table, top-down product photo",
        expected_count=7,
        expected_count_target="objects",
        expected_count_object="coin",
    ),
    PromptCase(
        name="anatomy_fullbody",
        prompt="full body portrait of a woman standing, visible hands and feet, natural pose, studio lighting",
    ),
    PromptCase(
        name="composition_wide",
        prompt="wide cinematic shot of a city street at sunset, balanced composition, readable depth",
        width=1216,
        height=704,
    ),
]


SUITE_PACKS: Dict[str, List[PromptCase]] = {
    "standard_v1": list(DEFAULT_SUITE),
    "top_contender_proxy_v1": list(DEFAULT_SUITE)
    + [
        PromptCase(
            name="attribute_binding",
            prompt="exactly 2 people: left person in blue shirt, right person in red shirt, medium shot, clear separation",
            expected_count=2,
            expected_count_target="people",
        ),
        PromptCase(
            name="long_typography",
            prompt='clean poster design with centered title text that says "FUTURE OF IMAGING 2026", high contrast, legible lettering',
            expected_text="FUTURE OF IMAGING 2026",
        ),
        PromptCase(
            name="object_count_windows",
            prompt="exactly 6 windows on a small house facade, daylight architectural photo",
            expected_count=6,
            expected_count_target="objects",
            expected_count_object="window",
        ),
        PromptCase(
            name="anatomy_hands",
            prompt="two hands holding a ceramic mug, natural fingers, realistic skin texture, sharp focus",
        ),
    ],
    "text_heavy_v1": [
        PromptCase(
            name="short_sign",
            prompt='storefront sign that says "OPEN NOW", clear lettering, high readability',
            expected_text="OPEN NOW",
        ),
        PromptCase(
            name="poster_title",
            prompt='event poster with title text that says "CITY LIGHT FEST", clean typography layout',
            expected_text="CITY LIGHT FEST",
        ),
        PromptCase(
            name="menu_board",
            prompt='chalkboard menu with heading text that says "DAILY SPECIALS", cafe interior',
            expected_text="DAILY SPECIALS",
        ),
    ],
    "count_stress_v1": [
        PromptCase(
            name="people_4",
            prompt="exactly 4 people standing in a row, full body, clear spacing",
            expected_count=4,
            expected_count_target="people",
        ),
        PromptCase(
            name="coins_9",
            prompt="exactly 9 coins arranged on plain background, top view",
            expected_count=9,
            expected_count_target="objects",
            expected_count_object="coin",
        ),
        PromptCase(
            name="candles_5",
            prompt="exactly 5 candles on a birthday cake, close-up photo",
            expected_count=5,
            expected_count_target="objects",
            expected_count_object="candle",
        ),
    ],
    "biz_visual_content_v1": [
        PromptCase(
            name="poster_layout_text",
            prompt='professional poster layout, headline text that says "SPRING DESIGN EXPO", subtitle text that says "APRIL 18", clean hierarchy',
            expected_text="SPRING DESIGN EXPO",
            width=896,
            height=1152,
        ),
        PromptCase(
            name="slide_like_infographic",
            prompt='clean presentation slide style infographic, 3 labeled sections, title text that says "MARKET OVERVIEW", business aesthetic',
            expected_text="MARKET OVERVIEW",
            width=1152,
            height=768,
        ),
        PromptCase(
            name="web_hero_banner",
            prompt='modern web hero section mockup with CTA button text that says "GET STARTED", clean spacing and alignment',
            expected_text="GET STARTED",
            width=1344,
            height=768,
        ),
        PromptCase(
            name="chart_like_composition",
            prompt='business chart-like visual with three distinct colored bars and caption text that says "Q4 RESULTS"',
            expected_text="Q4 RESULTS",
            expected_count=3,
            expected_count_target="objects",
            expected_count_object="bar",
            width=1024,
            height=768,
        ),
    ],
}


def _load_suite(path: Optional[Path], suite_pack: str = "standard_v1") -> List[PromptCase]:
    if path is None:
        return list(SUITE_PACKS.get((suite_pack or "standard_v1").strip().lower(), DEFAULT_SUITE))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("suite JSON must be a list of objects")
    out: List[PromptCase] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        out.append(
            PromptCase(
                name=str(row.get("name", "case")).strip() or "case",
                prompt=str(row.get("prompt", "")).strip(),
                expected_text=str(row.get("expected_text", "") or ""),
                expected_count=int(row.get("expected_count", 0) or 0),
                expected_count_target=str(row.get("expected_count_target", "auto") or "auto"),
                expected_count_object=str(row.get("expected_count_object", "") or ""),
                width=int(row.get("width", 1024) or 1024),
                height=int(row.get("height", 1024) or 1024),
            )
        )
    return [c for c in out if c.prompt]


def _collect_checkpoints(explicit: List[str], compare_to_dir: str) -> List[str]:
    ckpts = [str(x).strip() for x in explicit if str(x).strip()]
    d = str(compare_to_dir or "").strip()
    if not d:
        return ckpts
    root = Path(d)
    if not root.is_dir():
        return ckpts
    exts = (".pt", ".pth", ".safetensors")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            ckpts.append(str(p))
    # Stable order + dedupe.
    out: List[str] = []
    seen = set()
    for c in sorted(ckpts):
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _parse_seeds(seed: int, seed_list: str) -> List[int]:
    toks = [t.strip() for t in str(seed_list or "").split(",") if t.strip()]
    if not toks:
        return [int(seed)]
    out: List[int] = []
    for t in toks:
        try:
            out.append(int(t))
        except Exception:
            continue
    return out or [int(seed)]


def _aggregate_model_scores(rows: List[float], robustness_penalty: float) -> Dict[str, float]:
    if not rows:
        return {"mean_composite": 0.0, "std_composite": 0.0, "robust_score": 0.0}
    mean_v = float(np.mean(rows))
    std_v = float(np.std(rows))
    robust = float(mean_v - max(0.0, float(robustness_penalty)) * std_v)
    return {"mean_composite": mean_v, "std_composite": std_v, "robust_score": robust}


def _composite_score(case: PromptCase, metrics: Dict[str, float]) -> float:
    vals: List[float] = []
    # Exposure is already [0,1] in our scorer.
    vals.append(float(np.clip(metrics.get("exposure_balance", 0.5), 0.0, 1.0)))
    # Edge sharpness needs normalization.
    edge_raw = float(metrics.get("edge_sharpness", 0.0))
    vals.append(float(np.clip(edge_raw / 400.0, 0.0, 1.0)))
    vals.append(float(np.clip(metrics.get("saturation_balance", 0.75), 0.0, 1.0)))
    if case.expected_text.strip():
        vals.append(float(np.clip(metrics.get("ocr_match", 0.0), 0.0, 1.0)))
    if int(case.expected_count) > 0:
        vals.append(float(np.clip(metrics.get("count_match", 0.0), 0.0, 1.0)))
    if "clip_similarity" in metrics:
        # CLIP logits are unbounded; squash to [0,1].
        x = float(metrics["clip_similarity"])
        vals.append(float(1.0 / (1.0 + np.exp(-x / 10.0))))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _score_image(case: PromptCase, image_path: Path, *, include_clip: bool, clip_model: str, device: str) -> Dict[str, float]:
    arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    out: Dict[str, float] = {
        "edge_sharpness": float(ttp.score_edge_sharpness(arr)),
        "exposure_balance": float(ttp.score_exposure_balance(arr)),
        "saturation_balance": float(ttp.score_saturation_balance(arr)),
    }
    if case.expected_text.strip():
        out["ocr_match"] = float(ttp.score_ocr_match(arr, case.expected_text))
    if int(case.expected_count) > 0:
        if (case.expected_count_target or "auto").lower().strip() == "objects":
            out["count_match"] = float(
                ttp.score_object_count_match(arr, case.expected_count, object_hint=case.expected_count_object)
            )
        else:
            out["count_match"] = float(ttp.score_people_count_match(arr, case.expected_count))
    if include_clip:
        out["clip_similarity"] = float(
            ttp.score_clip_similarity([arr], case.prompt, device=device, model_id=clip_model)[0]
        )
    out["composite"] = _composite_score(case, out)
    return out


def _run_sample(
    sample_py: Path,
    ckpt: Path,
    out_png: Path,
    case: PromptCase,
    args: argparse.Namespace,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(sample_py),
        "--ckpt",
        str(ckpt),
        "--prompt",
        case.prompt,
        "--out",
        str(out_png),
        "--preset",
        str(args.preset),
        "--device",
        str(args.device),
        "--width",
        str(case.width),
        "--height",
        str(case.height),
        "--steps",
        str(int(args.steps)),
        "--seed",
        str(int(args.seed)),
        "--num",
        str(int(args.num)),
        "--pick-best",
        str(args.pick_best),
    ]
    if case.expected_text.strip():
        cmd.extend(["--expected-text", case.expected_text])
    if int(case.expected_count) > 0:
        cmd.extend(["--expected-count", str(int(case.expected_count))])
        cmd.extend(["--expected-count-target", str(case.expected_count_target)])
        if case.expected_count_object.strip():
            cmd.extend(["--expected-count-object", case.expected_count_object.strip()])
    if bool(args.auto_expected_text):
        cmd.append("--auto-expected-text")
    else:
        cmd.append("--no-auto-expected-text")
    for tok in args.sample_arg:
        cmd.append(str(tok))
    subprocess.run(cmd, check=True)


def _write_preference_jsonl(
    out_path: Path,
    rows: List[Dict[str, Any]],
    *,
    min_margin: float,
    max_pairs_per_case: int,
) -> int:
    # Local import avoids hard dependency when users only need benchmarking.
    from scripts.tools.training.mine_preference_pairs import mine_pairs

    pairs = mine_pairs(
        rows,
        min_margin=float(min_margin),
        max_pairs_per_case=int(max_pairs_per_case),
        require_existing_files=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(pairs)


def _failure_tags(case: PromptCase, metrics: Dict[str, float], *, threshold: float) -> List[str]:
    tags: List[str] = []
    comp = float(metrics.get("composite", 0.0) or 0.0)
    if comp < float(threshold):
        tags.append("low_composite")
    if case.expected_text.strip():
        if float(metrics.get("ocr_match", 0.0) or 0.0) < 0.7:
            tags.append("text_rendering")
    if int(case.expected_count) > 0:
        if float(metrics.get("count_match", 0.0) or 0.0) < 0.7:
            tags.append("counting")
    if float(metrics.get("exposure_balance", 0.5) or 0.5) < 0.6:
        tags.append("exposure")
    if float(metrics.get("saturation_balance", 0.5) or 0.5) < 0.6:
        tags.append("oversaturation")
    edge_norm = float(np.clip(float(metrics.get("edge_sharpness", 0.0) or 0.0) / 400.0, 0.0, 1.0))
    if edge_norm < 0.35:
        tags.append("blur")
    return tags


def _write_hardcases_jsonl(
    out_path: Path,
    rows: List[Dict[str, Any]],
    cases_by_name: Dict[str, PromptCase],
    *,
    threshold: float,
    max_rows: int,
) -> int:
    hard_rows: List[Dict[str, Any]] = []
    for r in rows:
        case_name = str(r.get("case", "") or "")
        case = cases_by_name.get(case_name)
        if case is None:
            continue
        metrics = {
            "composite": float(r.get("composite", 0.0) or 0.0),
            "ocr_match": float(r.get("ocr_match", 0.0) or 0.0),
            "count_match": float(r.get("count_match", 0.0) or 0.0),
            "exposure_balance": float(r.get("exposure_balance", 0.0) or 0.0),
            "saturation_balance": float(r.get("saturation_balance", 0.0) or 0.0),
            "edge_sharpness": float(r.get("edge_sharpness", 0.0) or 0.0),
        }
        tags = _failure_tags(case, metrics, threshold=float(threshold))
        if not tags:
            continue
        hard_rows.append(
            {
                "model": str(r.get("model", "") or ""),
                "case": case_name,
                "seed": int(r.get("seed", 0) or 0),
                "prompt": str(r.get("prompt", "") or ""),
                "image_path": str(r.get("output", "") or ""),
                "composite": float(metrics["composite"]),
                "failure_tags": tags,
            }
        )
    hard_rows.sort(key=lambda x: float(x.get("composite", 0.0)))
    if int(max_rows) > 0:
        hard_rows = hard_rows[: int(max_rows)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in hard_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(hard_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark checkpoint(s) on a prompt suite with quality metrics.")
    ap.add_argument("--ckpt", nargs="*", default=[], help="One or more checkpoint paths to compare.")
    ap.add_argument(
        "--compare-to-dir",
        type=str,
        default="",
        help="Recursively include all .pt/.pth/.safetensors checkpoints in a directory.",
    )
    ap.add_argument("--suite-json", type=str, default="", help="Optional custom suite JSON list.")
    ap.add_argument(
        "--suite-pack",
        type=str,
        default="standard_v1",
        choices=sorted(list(SUITE_PACKS.keys())),
        help="Built-in suite pack when --suite-json is not provided.",
    )
    ap.add_argument("--out-dir", type=str, default="benchmark_suite")
    ap.add_argument("--preset", type=str, default="sdxl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-list", type=str, default="", help="Comma-separated seeds for robustness runs (e.g. 42,123,999).")
    ap.add_argument(
        "--robustness-penalty",
        type=float,
        default=0.15,
        help="Leaderboard penalty weight on score std across seed runs.",
    )
    ap.add_argument("--num", type=int, default=3, help="Candidates per case before pick-best.")
    ap.add_argument("--pick-best", type=str, default="auto")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--include-clip-score", action="store_true")
    ap.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--auto-expected-text", action="store_true", default=True)
    ap.add_argument("--no-auto-expected-text", action="store_false", dest="auto_expected_text")
    ap.add_argument("--sample-arg", action="append", default=[], help="Extra token passed to sample.py; repeat as needed.")
    ap.add_argument(
        "--export-preference-jsonl",
        type=str,
        default="",
        help="If set, mine benchmark rows into DPO preference JSONL at this path.",
    )
    ap.add_argument("--preference-min-margin", type=float, default=0.08)
    ap.add_argument("--preference-max-pairs-per-case", type=int, default=2)
    ap.add_argument(
        "--export-hardcases-jsonl",
        type=str,
        default="",
        help="If set, export low-scoring failure-tagged rows for targeted retraining/curation.",
    )
    ap.add_argument("--hardcase-threshold", type=float, default=0.60, help="Composite threshold for hard-case tagging.")
    ap.add_argument("--hardcase-max-rows", type=int, default=200, help="Maximum hard-case rows to export.")
    args = ap.parse_args()

    checkpoints = _collect_checkpoints(args.ckpt, args.compare_to_dir)
    if not checkpoints:
        raise SystemExit("Provide --ckpt and/or --compare-to-dir with checkpoints.")
    suite_path = Path(args.suite_json) if str(args.suite_json).strip() else None
    cases = _load_suite(suite_path, suite_pack=str(args.suite_pack))
    if not cases:
        raise SystemExit("No prompt cases found.")
    cases_by_name = {c.name: c for c in cases}

    repo_root = Path(__file__).resolve().parents[2]
    sample_py = repo_root / "sample.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(int(args.seed), str(args.seed_list))
    results: List[Dict[str, Any]] = []
    leaderboard: Dict[str, List[float]] = {}

    for ckpt_s in checkpoints:
        ckpt = Path(ckpt_s)
        tag = ckpt.stem
        model_dir = out_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(cases):
            for seed in seeds:
                out_png = model_dir / f"{i:03d}_{case.name}_s{int(seed)}.png"
                if not (args.skip_existing and out_png.is_file()):
                    print(f"[benchmark_suite] {tag} :: {case.name} :: seed={int(seed)}")
                    args.seed = int(seed)
                    _run_sample(sample_py, ckpt, out_png, case, args)
                met = _score_image(
                    case,
                    out_png,
                    include_clip=bool(args.include_clip_score),
                    clip_model=str(args.clip_model),
                    device=str(args.device),
                )
                row = {
                    "model": tag,
                    "checkpoint": str(ckpt),
                    "case": case.name,
                    "seed": int(seed),
                    "output": str(out_png),
                    "prompt": case.prompt,
                    **met,
                }
                results.append(row)
                leaderboard.setdefault(tag, []).append(float(met.get("composite", 0.0)))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lb_rows = []
    for model_tag, vals in leaderboard.items():
        agg = _aggregate_model_scores(vals, robustness_penalty=float(args.robustness_penalty))
        lb_rows.append(
            {
                "model": model_tag,
                "mean_composite": float(agg["mean_composite"]),
                "std_composite": float(agg["std_composite"]),
                "robust_score": float(agg["robust_score"]),
                "cases": len(vals),
            }
        )
    lb_rows.sort(key=lambda x: x["robust_score"], reverse=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(lb_rows, indent=2), encoding="utf-8")
    with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mean_composite", "std_composite", "robust_score", "cases"])
        w.writeheader()
        w.writerows(lb_rows)
    if str(args.export_preference_jsonl).strip():
        pref_path = Path(str(args.export_preference_jsonl).strip())
        n_pairs = _write_preference_jsonl(
            pref_path,
            results,
            min_margin=float(args.preference_min_margin),
            max_pairs_per_case=int(args.preference_max_pairs_per_case),
        )
        print(f"[benchmark_suite] wrote {n_pairs} preference pairs: {pref_path}")
    if str(args.export_hardcases_jsonl).strip():
        hard_path = Path(str(args.export_hardcases_jsonl).strip())
        n_hard = _write_hardcases_jsonl(
            hard_path,
            results,
            cases_by_name,
            threshold=float(args.hardcase_threshold),
            max_rows=int(args.hardcase_max_rows),
        )
        print(f"[benchmark_suite] wrote {n_hard} hard-case rows: {hard_path}")
    print(f"[benchmark_suite] wrote {out_dir / 'results.json'}")
    print(f"[benchmark_suite] wrote {out_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
