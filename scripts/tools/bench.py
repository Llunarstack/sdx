#!/usr/bin/env python3
"""Checkpoint benchmarks and sample.py sweeps: suite, history, prompts, seeds, gallery.

    python -m scripts.tools bench suite --ckpt results/a/best.pt results/b/best.pt --preset sdxl
    python -m scripts.tools bench history --leaderboard benchmark_out/leaderboard.json
    python -m scripts.tools bench prompts --ckpt best.pt --preset sdxl --op-mode portrait
    python -m scripts.tools bench seeds --ckpt best.pt --prompt "..." --rows 2 --cols 4
    python -m scripts.tools bench gallery --ckpt best.pt --preset anime

Old names still work: ``benchmark_suite``, ``benchmark_history``, ``eval_prompts``, ``seed_explorer``,
``make_gallery``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SAMPLE_PY = REPO / "sample.py"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- suite (benchmark_suite)


@dataclass(slots=True)
class PromptCase:
    name: str
    prompt: str
    expected_text: str = ""
    expected_count: int = 0
    expected_count_target: str = "auto"  # auto|people|objects
    expected_count_object: str = ""
    width: int = 1024
    height: int = 1024
    prompt_layout: str = ""  # optional JSON path for --prompt-layout (penta/triple CLIP routing)


DEFAULT_SUITE: list[PromptCase] = [
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


SUITE_PACKS: dict[str, list[PromptCase]] = {
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
            prompt_layout="examples/prompt_layout.example.json",
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


def _load_suite(path: Path | None, suite_pack: str = "standard_v1") -> list[PromptCase]:
    if path is None:
        return list(SUITE_PACKS.get((suite_pack or "standard_v1").strip().lower(), DEFAULT_SUITE))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("suite JSON must be a list of objects")
    out = [
        PromptCase(
            name=str(row.get("name", "case")).strip() or "case",
            prompt=str(row.get("prompt", "")).strip(),
            expected_text=str(row.get("expected_text", "") or ""),
            expected_count=int(row.get("expected_count", 0) or 0),
            expected_count_target=str(row.get("expected_count_target", "auto") or "auto"),
            expected_count_object=str(row.get("expected_count_object", "") or ""),
            width=int(row.get("width", 1024) or 1024),
            height=int(row.get("height", 1024) or 1024),
            prompt_layout=str(row.get("prompt_layout", "") or row.get("prompt_layout_path", "") or ""),
        )
        for row in data
        if isinstance(row, dict)
    ]
    return [c for c in out if c.prompt]


def _collect_checkpoints(explicit: list[str], compare_to_dir: str) -> list[str]:
    ckpts = [str(x).strip() for x in explicit if str(x).strip()]
    d = str(compare_to_dir or "").strip()
    if not d or not Path(d).is_dir():
        return ckpts
    ckpts += [str(p) for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in (".pt", ".pth", ".safetensors")]
    return sorted(set(ckpts))


def _parse_seeds(seed: int, seed_list: str) -> list[int]:
    out: list[int] = []
    for t in str(seed_list or "").split(","):
        try:
            out.append(int(t.strip()))
        except ValueError:
            continue
    return out or [int(seed)]


def _aggregate_model_scores(rows: list[float], robustness_penalty: float) -> dict[str, float]:
    if not rows:
        return {"mean_composite": 0.0, "std_composite": 0.0, "robust_score": 0.0}
    mean_v = float(np.mean(rows))
    std_v = float(np.std(rows))
    robust = float(mean_v - max(0.0, float(robustness_penalty)) * std_v)
    return {"mean_composite": mean_v, "std_composite": std_v, "robust_score": robust}


def _composite_score(case: PromptCase, metrics: dict[str, float]) -> float:
    vals: list[float] = [
        float(np.clip(metrics.get("exposure_balance", 0.5), 0.0, 1.0)),
        float(np.clip(float(metrics.get("edge_sharpness", 0.0)) / 400.0, 0.0, 1.0)),
        float(np.clip(metrics.get("saturation_balance", 0.75), 0.0, 1.0)),
    ]
    if case.expected_text.strip():
        vals.append(float(np.clip(metrics.get("ocr_match", 0.0), 0.0, 1.0)))
    if int(case.expected_count) > 0:
        vals.append(float(np.clip(metrics.get("count_match", 0.0), 0.0, 1.0)))
    for key in ("natural", "art"):
        if key in metrics:
            vals.append(float(np.clip(metrics[key], 0.0, 1.0)))
    if "clip_similarity" in metrics:
        # CLIP logits are unbounded; squash to [0,1].
        vals.append(float(1.0 / (1.0 + np.exp(-float(metrics["clip_similarity"]) / 10.0))))
    return float(sum(vals) / len(vals))


def _score_image(
    case: PromptCase, image_path: Path, *, include_clip: bool, clip_model: str, device: str
) -> dict[str, float]:
    from PIL import Image
    from utils.quality import test_time_pick as ttp
    from utils.quality.critics import ai_look_score, score_art, score_oversaturation

    arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    sat = score_oversaturation(arr)
    out: dict[str, float] = {
        "edge_sharpness": float(ttp.score_edge_sharpness(arr)),
        "exposure_balance": float(ttp.score_exposure_balance(arr)),
        "saturation_balance": 0.5 if sat is None else float(sat),
        "natural": ai_look_score(arr, case.prompt).natural,
        "art": score_art(arr, prompt=case.prompt).score,
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


def _run_sample(ckpt: Path, out_png: Path, case: PromptCase, args: argparse.Namespace) -> None:
    cmd: list[str] = [
        sys.executable,
        str(SAMPLE_PY),
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
        cmd += ["--expected-text", case.expected_text]
    if int(case.expected_count) > 0:
        cmd += ["--expected-count", str(int(case.expected_count))]
        cmd += ["--expected-count-target", str(case.expected_count_target)]
        if case.expected_count_object.strip():
            cmd += ["--expected-count-object", case.expected_count_object.strip()]
    cmd.append("--auto-expected-text" if args.auto_expected_text else "--no-auto-expected-text")
    if str(args.local_rag_jsonl or "").strip():
        cmd += ["--local-rag-jsonl", str(args.local_rag_jsonl).strip()]
    if str(args.pick_vit_ckpt or "").strip():
        cmd += ["--pick-vit-ckpt", str(args.pick_vit_ckpt).strip()]
    pl = str(case.prompt_layout or "").strip()
    if pl:
        layout_path = Path(pl)
        if not layout_path.is_file():
            layout_path = REPO / pl
        if layout_path.is_file():
            cmd += ["--prompt-layout", str(layout_path)]
    cmd += [str(tok) for tok in args.sample_arg]
    subprocess.run(cmd, check=True)


def _write_preference_jsonl(
    out_path: Path,
    rows: list[dict[str, Any]],
    *,
    min_margin: float,
    max_pairs_per_case: int,
    vit_ckpt: str = "",
    vit_mine: bool = False,
    device: str = "cuda",
) -> int:
    if vit_mine and vit_ckpt.strip():
        from utils.agentic.quality_loop import ViTMineConfig, mine_vit_preference_pairs

        pairs = mine_vit_preference_pairs(
            rows,
            ViTMineConfig(
                vit_ckpt=vit_ckpt.strip(),
                min_margin=float(min_margin),
                max_pairs_per_case=int(max_pairs_per_case),
                device=device,
            ),
        )
    else:
        from scripts.tools.superior import mine_pairs

        pairs = mine_pairs(
            rows,
            min_margin=float(min_margin),
            max_pairs_per_case=int(max_pairs_per_case),
            require_existing_files=False,
        )
    _write_jsonl(out_path, pairs)
    return len(pairs)


def _failure_tags(case: PromptCase, metrics: dict[str, float], *, threshold: float) -> list[str]:
    tags: list[str] = []
    if float(metrics.get("composite", 0.0) or 0.0) < float(threshold):
        tags.append("low_composite")
    if case.expected_text.strip() and float(metrics.get("ocr_match", 0.0) or 0.0) < 0.7:
        tags.append("text_rendering")
    if int(case.expected_count) > 0 and float(metrics.get("count_match", 0.0) or 0.0) < 0.7:
        tags.append("counting")
    if float(metrics.get("exposure_balance", 0.5) or 0.5) < 0.6:
        tags.append("exposure")
    if float(metrics.get("saturation_balance", 0.5) or 0.5) < 0.6:
        tags.append("oversaturation")
    if float(np.clip(float(metrics.get("edge_sharpness", 0.0) or 0.0) / 400.0, 0.0, 1.0)) < 0.35:
        tags.append("blur")
    if float(metrics.get("natural", 1.0)) < 0.5:
        tags.append("ai_look")
    return tags


def _write_hardcases_jsonl(
    out_path: Path,
    rows: list[dict[str, Any]],
    cases_by_name: dict[str, PromptCase],
    *,
    threshold: float,
    max_rows: int,
) -> int:
    hard_rows: list[dict[str, Any]] = []
    for r in rows:
        case_name = str(r.get("case", "") or "")
        case = cases_by_name.get(case_name)
        if case is None:
            continue
        metrics = {
            k: float(r.get(k, 0.0) or 0.0)
            for k in (
                "composite",
                "ocr_match",
                "count_match",
                "exposure_balance",
                "saturation_balance",
                "edge_sharpness",
            )
        }
        metrics["natural"] = float(r.get("natural", 1.0))
        tags = _failure_tags(case, metrics, threshold=float(threshold))
        if tags:
            hard_rows.append(
                {
                    "model": str(r.get("model", "") or ""),
                    "case": case_name,
                    "seed": int(r.get("seed", 0) or 0),
                    "prompt": str(r.get("prompt", "") or ""),
                    "image_path": str(r.get("output", "") or ""),
                    "composite": metrics["composite"],
                    "failure_tags": tags,
                }
            )
    hard_rows.sort(key=lambda x: float(x.get("composite", 0.0)))
    if int(max_rows) > 0:
        hard_rows = hard_rows[: int(max_rows)]
    _write_jsonl(out_path, hard_rows)
    return len(hard_rows)


def _suite_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bench suite", description="Benchmark checkpoint(s) on a prompt suite with quality metrics."
    )
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
        choices=sorted(SUITE_PACKS),
        help="Built-in suite pack when --suite-json is not provided.",
    )
    ap.add_argument("--out-dir", type=str, default="benchmark_suite")
    ap.add_argument("--preset", type=str, default="sdxl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seed-list", type=str, default="", help="Comma-separated seeds for robustness runs (e.g. 42,123,999)."
    )
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
    ap.add_argument(
        "--sample-arg", action="append", default=[], help="Extra token passed to sample.py; repeat as needed."
    )
    ap.add_argument(
        "--local-rag-jsonl",
        type=str,
        default="",
        help="Pass --local-rag-jsonl to sample.py for grounded benchmark prompts.",
    )
    ap.add_argument(
        "--pick-vit-ckpt",
        type=str,
        default="",
        help="Pass --pick-vit-ckpt to sample.py; also used for ViT scoring when --vit-score is set.",
    )
    ap.add_argument(
        "--vit-score",
        action="store_true",
        help="Add vit_quality/vit_reward/blended_score to each result row (requires --pick-vit-ckpt).",
    )
    ap.add_argument(
        "--vit-mine-preferences",
        action="store_true",
        help="When exporting preferences, use ViT-weighted mining (requires --pick-vit-ckpt).",
    )
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
        help="If set, export low-scoring failure-tagged rows for targeted retr/curation.",
    )
    ap.add_argument("--hardcase-threshold", type=float, default=0.60, help="Composite threshold for hard-case tagging.")
    ap.add_argument("--hardcase-max-rows", type=int, default=200, help="Maximum hard-case rows to export.")
    args = ap.parse_args(argv)

    checkpoints = _collect_checkpoints(args.ckpt, args.compare_to_dir)
    if not checkpoints:
        raise SystemExit("Provide --ckpt and/or --compare-to-dir with checkpoints.")
    suite_path = Path(args.suite_json) if str(args.suite_json).strip() else None
    cases = _load_suite(suite_path, suite_pack=str(args.suite_pack))
    if not cases:
        raise SystemExit("No prompt cases found.")
    cases_by_name = {c.name: c for c in cases}

    from utils.modeling.encoders import checkpoint_text_stack, text_encoder_mode_label

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(int(args.seed), str(args.seed_list))
    use_vit = bool(args.vit_score) and str(args.pick_vit_ckpt or "").strip()
    results: list[dict[str, Any]] = []
    leaderboard: dict[str, list[float]] = {}

    for ckpt_s in checkpoints:
        ckpt = Path(ckpt_s)
        tag = ckpt.stem
        te_mode, te_fusion = checkpoint_text_stack(str(ckpt))
        print(
            f"[benchmark_suite] checkpoint {tag}: text_encoder_mode={te_mode} "
            f"({text_encoder_mode_label(te_mode)}), fusion_weights={te_fusion}"
        )
        model_dir = out_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        for i, case in enumerate(cases):
            for seed in seeds:
                out_png = model_dir / f"{i:03d}_{case.name}_s{int(seed)}.png"
                if not (args.skip_existing and out_png.is_file()):
                    print(f"[benchmark_suite] {tag} :: {case.name} :: seed={int(seed)}")
                    args.seed = int(seed)
                    _run_sample(ckpt, out_png, case, args)
                met = _score_image(
                    case,
                    out_png,
                    include_clip=bool(args.include_clip_score),
                    clip_model=str(args.clip_model),
                    device=str(args.device),
                )
                if use_vit:
                    from utils.agentic.quality_loop import blended_reward, score_image_vit

                    _q, vit_r = score_image_vit(
                        out_png, case.prompt, vit_ckpt=str(args.pick_vit_ckpt), device=str(args.device)
                    )
                    met["vit_quality"] = _q
                    met["vit_reward"] = vit_r
                    met["blended_score"] = blended_reward(float(met.get("composite", 0)), vit_r)
                results.append(
                    {
                        "model": tag,
                        "checkpoint": str(ckpt),
                        "text_encoder_mode": te_mode,
                        "text_encoder_fusion": te_fusion,
                        "case": case.name,
                        "seed": int(seed),
                        "output": str(out_png),
                        "prompt": case.prompt,
                        **met,
                    }
                )
                leaderboard.setdefault(tag, []).append(float(met.get("composite", 0.0)))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lb_rows = [
        {"model": tag, **_aggregate_model_scores(vals, float(args.robustness_penalty)), "cases": len(vals)}
        for tag, vals in leaderboard.items()
    ]
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
            vit_ckpt=str(args.pick_vit_ckpt or ""),
            vit_mine=bool(args.vit_mine_preferences),
            device=str(args.device),
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
    return 0


# ---------------------------------------------------------------- history


def _history_main(argv: list[str]) -> int:
    from utils.generation.sample_prompt import append_benchmark_history

    p = argparse.ArgumentParser(
        prog="bench history", description="Append benchmark_suite leaderboard snapshots to rolling history."
    )
    p.add_argument("--leaderboard", type=Path, default=Path("benchmark_out/leaderboard.json"))
    p.add_argument("--history", type=Path, default=Path("benchmark_out/history.json"))
    args = p.parse_args(argv)
    if not args.leaderboard.is_file():
        p.error(f"leaderboard not found: {args.leaderboard}")
    entry = append_benchmark_history(args.leaderboard, args.history)
    print(f"Appended snapshot at {entry['timestamp']} -> {args.history}")
    return 0


# ---------------------------------------------------------------- prompts (eval_prompts)

EVAL_PROMPTS: dict[str, str] = {
    "portrait_real": "portrait of a woman, soft window light, shallow depth of field, sharp eyes, 50mm lens",
    "fullbody_char": "1girl, standing, long dress, legs, shoes, full body, outdoors, sunset",
    "3d_car": "3d render of a red car parked next to a blue house, sunset, detailed reflections",
    "interior_room": "modern living room interior, sofa, coffee table, large window, natural light, detailed background",
    "exterior_city": "street view, urban city at night, neon signs, rain, reflections",
    "anime_char": "1girl, school uniform, anime style, dynamic lighting, detailed hair",
}


def _preset_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.preset:
        out += ["--preset", args.preset]
    if args.op_mode:
        out += ["--op-mode", args.op_mode]
    return out


def _prompts_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bench prompts",
        description="Run a fixed canonical prompt set through sample.py for quick qualitative regression checks. "
        "Unknown flags pass through to sample.py.",
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--op-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="eval_prompts")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None, help="Override preset steps for evaluation")
    args, unknown = ap.parse_known_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    for key, prompt in EVAL_PROMPTS.items():
        out_path = out_dir / f"{key}.png"
        cli_args = ["--ckpt", args.ckpt, "--prompt", prompt, "--out", str(out_path)]
        cli_args += ["--width", str(args.width), "--height", str(args.height), "--num", "1", *_preset_args(args)]
        if args.steps is not None:
            cli_args += ["--steps", str(args.steps)]
        print(f"[eval_prompts] Running sample.py for '{key}'")
        subprocess.run([sys.executable, str(SAMPLE_PY), *cli_args, *unknown], check=True)
        results.append({"name": key, "prompt": prompt, "path": str(out_path)})

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved eval index: {index_path}")
    return 0


# ---------------------------------------------------------------- seeds (seed_explorer)


def save_grid(images: list[Path], out_path: Path, *, cols: int, rows: int | None = None) -> None:
    """Paste readable images row-major into a ``cols``-wide grid (cell size of the first image)."""
    from PIL import Image

    pil_images = []
    for p in images:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except OSError:
            continue
    if not pil_images:
        print("No images to grid.")
        return
    rows = rows or -(-len(pil_images) // cols)
    w, h = pil_images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for idx, img in enumerate(pil_images[: rows * cols]):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved grid: {out_path} ({len(pil_images)} images)")


def _seeds_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bench seeds",
        description="Explore seeds for a prompt: one sample.py run per seed, plus grid.png and seeds.json. "
        "Unknown flags pass through to sample.py.",
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--negative-prompt", type=str, default="")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--op-mode", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="seed_explorer")
    args, unknown = ap.parse_known_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.rows * args.cols)]
    img_paths: list[Path] = []
    for seed in seeds:
        out_path = out_dir / f"seed_{seed}.png"
        cli_args = ["--ckpt", args.ckpt, "--prompt", args.prompt, "--negative-prompt", args.negative_prompt]
        cli_args += ["--seed", str(seed), "--out", str(out_path), "--num", "1", *_preset_args(args)]
        print(f"[seed_explorer] Running sample.py with seed={seed}")
        subprocess.run([sys.executable, str(SAMPLE_PY), *cli_args, *unknown], check=True)
        img_paths.append(out_path)

    save_grid(img_paths, out_dir / "grid.png", cols=args.cols, rows=args.rows)
    (out_dir / "seeds.json").write_text(json.dumps({"seeds": seeds}, indent=2), encoding="utf-8")
    print(f"Saved seeds: {out_dir / 'seeds.json'}")
    return 0


# ---------------------------------------------------------------- gallery (make_gallery)

GALLERY_PROMPTS = [
    "cinematic portrait of a young woman, soft window light, shallow depth of field, film grain",
    "hero character full body, dynamic action pose, city rooftop at night, anime style",
    "photoreal landscape, misty mountain valley at dawn, golden hour, ultra detailed",
    "book illustration, cozy cottage interior, warm candlelight, watercolor style",
    "sci-fi concept art, alien megastructure, volumetric fog, dramatic lighting",
    "character sheet, front and side view, clean lineart, white background",
]
GALLERY_NEGATIVE = "blurry, low quality, watermark, text, oversaturated, deformed"


def _gallery_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="bench gallery",
        description="Generate a gallery grid (README gallery) from a trained SDX checkpoint.",
    )
    ap.add_argument("--ckpt", required=True, help="Path to SDX checkpoint.")
    ap.add_argument("--prompts-file", default="", help="Text file with one prompt per line.")
    ap.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "balanced", "photoreal", "anime", "illustration", "aggressive"],
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cols", type=int, default=3, help="Grid columns.")
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "docs" / "assets" / "gallery"),
        help="Output directory for individual images and grid.",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        prompts = GALLERY_PROMPTS

    print(f"Generating {len(prompts)} images → {out_dir}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Preset     : {args.preset}  Steps: {args.steps}  CFG: {args.cfg}\n")

    generated: list[Path] = []
    for i, prompt in enumerate(prompts):
        slug = prompt[:40].lower().replace(" ", "_").replace(",", "").replace(".", "")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        out = out_dir / f"{i:02d}_{slug}.png"
        print(f"[{i + 1}/{len(prompts)}] {prompt[:60]}…")
        cmd = [sys.executable, str(SAMPLE_PY), "--ckpt", args.ckpt, "--prompt", prompt]
        cmd += ["--negative-prompt", GALLERY_NEGATIVE, "--holy-grail-preset", args.preset]
        cmd += ["--steps", str(args.steps), "--cfg-scale", str(args.cfg), "--width", str(args.width)]
        cmd += ["--height", str(args.height), "--seed", str(args.seed + i), "--device", args.device, "--out", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            generated.append(out)
            print(f"  → {out.name}")
        else:
            print(f"  FAILED: {result.stderr[-300:] if result.stderr else '(no stderr)'}")
            print("  → skipped (error)")

    if not generated:
        print("No images generated — check checkpoint path and sample.py errors.")
        return 1
    grid_path = out_dir / "gallery_grid.png"
    save_grid(generated, grid_path, cols=args.cols)
    print(f"\nGallery complete: {len(generated)}/{len(prompts)} images")
    print(f"Grid: {grid_path}\n")
    print("Add to README.md:")
    print("  ![Gallery](docs/assets/gallery/gallery_grid.png)")
    return 0


COMMANDS = {
    "suite": _suite_main,
    "history": _history_main,
    "prompts": _prompts_main,
    "seeds": _seeds_main,
    "gallery": _gallery_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
