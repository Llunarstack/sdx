"""``sample.py`` orchestration: resolve flags and soft defaults, then run each generation phase in order.

Precedence for every setting: flags the user typed > explicit ``--preset`` / ``--op-mode`` /
``--holy-grail-preset`` > the quality policy (``config.defaults.quality_policy``) > the holy-grail preset
the policy picks > parser defaults.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from utils.terminal import configure_stdio_for_console

configure_stdio_for_console()

from utils.generation.sample_cli_parser import build_sample_parser
from utils.generation.sample_conditioning import (
    conditioning_image_paths,
    encode_conditioning,
    prepare_init_latents,
)
from utils.generation.sample_helpers import SAMPLE_PY
from utils.generation.sample_output import (
    after_save,
    capture_finish,
    choose_best,
    decode_images,
    postprocess,
    save_targets,
)
from utils.generation.sample_prompt import build_prompt_plan, compose_prompt_additions, compose_user_prompt
from utils.generation.sample_sampling import export_attention, sample_latents
from utils.generation.sample_setup import load_models
from utils.quality.camera_isp import save_rgb
from utils.safety.minor_guard import require_image_allowed


def explicit_flags(parser: Any, argv: list[str]) -> frozenset[str]:
    """Destinations of the options the user typed (``--flag``, ``--flag=v``, unique ``--fl`` prefixes)."""
    options = {opt: action.dest for action in parser._actions for opt in action.option_strings}
    dests = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        dest = options.get(flag)
        if dest is None:
            hits = {d for opt, d in options.items() if opt.startswith(flag)}
            dest = hits.pop() if len(hits) == 1 else None
        if dest is not None:
            dests.add(dest)
    return frozenset(dests)


def keep_explicit(args: Any, apply: Any, *apply_args: Any) -> None:
    """Run a soft-default applier, then restore every flag the user typed (text it only extended is kept)."""
    typed = {k: getattr(args, k) for k in getattr(args, "_explicit_flags", ()) if hasattr(args, k)}
    apply(args, *apply_args)
    for k, v in typed.items():
        new = getattr(args, k)
        if isinstance(v, str) and v and isinstance(new, str) and new.startswith(v):
            continue
        setattr(args, k, v)


def _expand_presets(args: Any) -> None:
    from config.defaults.model_presets import apply_op_mode_to_args, apply_preset_to_args
    from diffusion.sampling import apply_holy_grail_preset_to_args, recommend_holy_grail_preset

    done = args.__dict__.setdefault("_expanded_presets", set())
    if getattr(args, "preset", None) and ("preset", args.preset) not in done:
        apply_preset_to_args(args, args.preset)
        done.add(("preset", args.preset))
    if getattr(args, "op_mode", None) and ("op_mode", args.op_mode) not in done:
        apply_op_mode_to_args(args, args.op_mode)
        done.add(("op_mode", args.op_mode))
    hg = str(getattr(args, "holy_grail_preset", None) or "").strip().lower()
    if hg and ("holy_grail", hg) not in done:
        name = hg
        if hg == "auto":
            name = recommend_holy_grail_preset(
                prompt=str(getattr(args, "prompt", "") or ""),
                style=str(getattr(args, "style", "") or ""),
                has_control=bool(getattr(args, "control_image", "") or getattr(args, "control", [])),
                has_lora=bool(getattr(args, "lora", [])),
            )
        apply_holy_grail_preset_to_args(args, name)
        done.add(("holy_grail", hg))


def expand_presets(args: Any) -> None:
    """Soft-apply ``--preset`` / ``--op-mode`` / ``--holy-grail-preset`` (``auto`` reads the prompt), once each."""
    keep_explicit(args, _expand_presets)


def parse_args(parser: Any, argv: list[str]) -> Any:
    args = parser.parse_args(argv)
    args._explicit_flags = explicit_flags(parser, argv)
    return args


def run_early_exit(args: Any, argv: list[str]) -> bool:
    """``--export-comfy-workflow`` and ``--auto-refine N`` (N child runs of ``sample.py``); True when handled."""
    if str(getattr(args, "export_comfy_workflow", "") or "").strip():
        from utils.generation.comfy_export import sample_args_to_comfy_workflow, write_comfy_workflow

        out = write_comfy_workflow(str(args.export_comfy_workflow).strip(), sample_args_to_comfy_workflow(vars(args)))
        print(f"Wrote Comfy workflow: {out}")
        return True
    n = int(getattr(args, "auto_refine", 0) or 0)
    if n <= 1:
        return False
    from utils.generation.sample_features import run_auto_refine_candidates, run_creative_refine

    stripped, skip = [], False
    for tok in argv:
        if skip or tok == "--auto-refine" or tok.startswith("--auto-refine="):
            skip = tok == "--auto-refine"
            continue
        stripped.append(tok)
    cmd = [sys.executable, str(SAMPLE_PY), *stripped]
    stem = str(Path(args.out).with_suffix("")) + "_ar"
    seed = int(getattr(args, "seed", 42) or 42)
    mutate = int(getattr(args, "creative_mutate", 0) or 0)
    if bool(getattr(args, "frontier_creative", False)) or mutate > 0:
        res = run_creative_refine(
            sample_cmd=cmd,
            base_prompt=str(getattr(args, "prompt", "") or ""),
            num_candidates=n,
            seed_base=seed,
            out_stem=stem,
            mutate_count=mutate if mutate > 0 else n,
            base_serendipity=float(getattr(args, "frontier_serendipity", 0.25) or 0.25),
        )
    else:
        res = run_auto_refine_candidates(sample_cmd=cmd, num_candidates=n, seed_base=seed, out_stem=stem)
    shutil.copy(res.outputs[res.best_index], args.out)
    best_prompt = res.prompts[res.best_index] if res.prompts else ""
    note = f" prompt={best_prompt[:80]!r}" if best_prompt else ""
    print(
        f"Auto-refine: kept candidate {res.best_index} (score={res.scores[res.best_index]:.3f}){note} -> {args.out}",
        file=sys.stderr,
    )
    return True


def _expand_explore_styles(args: Any) -> None:
    if getattr(args, "explore_styles_insane", False):
        args.explore_styles = True
        args.style_inventor_mode = "apocalypse"
        args.style_chaos_level = max(float(getattr(args, "style_chaos_level", 0) or 0), 0.95)
        args.invent_styles = max(int(getattr(args, "invent_styles", 0) or 0), 4)
        clauses = str(getattr(args, "prompt_clauses", "") or "").strip()
        insane = "style.chaos,style.surreal,style.apocalypse"
        args.prompt_clauses = f"{clauses},{insane}".strip(",") if clauses else insane
        print("Explore-styles-insane: apocalypse mode engaged.", file=sys.stderr)
    if getattr(args, "explore_styles", False):
        if int(getattr(args, "invent_styles", 0) or 0) < 1:
            args.invent_styles = 3
        if str(getattr(args, "pick_best", "none") or "none").strip().lower() in ("none", ""):
            args.pick_best = "combo"
        print(
            f"Explore-styles: inventing {args.invent_styles} genomes "
            f"(mode={getattr(args, 'style_inventor_mode', 'normal')}, "
            f"chaos={float(getattr(args, 'style_chaos_level', 0) or 0):.2f}); "
            f"use --style-genome-index 0..{args.invent_styles - 1} per run or scripts.tools.explore_styles",
            file=sys.stderr,
        )


def resolve_quality_defaults(args: Any, report: dict) -> None:
    """User-requested expansions, art direction, then the quality policy and the presets it picks."""
    keep_explicit(args, _expand_explore_styles)

    from utils.generation.quality_stack import apply_best_model_policy
    from utils.prompt.art_direction import apply_art_direction_to_args

    apply_art_direction_to_args(args)
    report["art_direction"] = args.art_direction_plan
    keep_explicit(args, apply_best_model_policy)
    expand_presets(args)
    if str(getattr(args, "human_made", "none") or "none").lower() not in ("none", "off", "0", ""):
        from utils.quality.human_made import apply_human_made_prompt_flags

        keep_explicit(args, apply_human_made_prompt_flags)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_sample_parser()
    args = parse_args(parser, argv)
    jobs = str(getattr(args, "jobs_jsonl", "") or "").strip()
    if not jobs:
        _generate(parser, args, argv)
        return
    # Batch mode: later flags win in argparse, so each job's prompt/seed/out override the base argv.
    # ponytail: models load with the first job's args, so prompt-picked LoRA-bank adapters stick for the
    # whole batch; pass --no-lora-bank (rsi_loop does) or run separate calls if adapters must vary.
    m = None
    for line in Path(jobs).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        job = json.loads(line)
        job_argv = [*argv, "--prompt", str(job["prompt"]), "--seed", str(int(job.get("seed", 42))), "--out", job["out"]]
        m = _generate(parser, parse_args(parser, job_argv), job_argv, m)


def _generate(parser: Any, args: Any, argv: list[str], m: Any = None) -> Any:
    """One generation; pass ``m`` from a previous call to skip reloading checkpoint and encoders."""
    report: dict = {
        "prompt": str(getattr(args, "prompt", "") or ""),
        "negative_prompt": str(getattr(args, "negative_prompt", "") or ""),
        "seed": int(getattr(args, "seed", 0) or 0),
        "steps": int(getattr(args, "steps", 0) or 0),
        "num": int(getattr(args, "num", 0) or 0),
    }
    expand_presets(args)
    if run_early_exit(args, argv):
        return m
    if float(getattr(args, "linear_attn_fraction", 0.0) or 0.0) > 0.0:
        from models.attention import set_linear_attention_fraction

        set_linear_attention_fraction(float(args.linear_attn_fraction))

    compose_user_prompt(args, parser)
    resolve_quality_defaults(args, report)
    extras = compose_prompt_additions(args)
    if m is None:
        m = load_models(args)
    else:
        import torch

        torch.manual_seed(int(args.seed))  # load_models seeds the first job; noise comes from the global RNG
    plan = build_prompt_plan(args, m, extras)
    cond = encode_conditioning(args, m, plan)
    init = prepare_init_latents(args, m, plan.num_gen)
    x0 = sample_latents(args, m, plan, cond, init, report)
    export_attention(args, m, cond, x0)

    images, out_size, fit = decode_images(args, m, x0)
    processed = postprocess(args, plan, images, out_size, fit)
    metric, best = choose_best(args, m, plan, processed, report)
    finals = capture_finish(args, plan, processed, report)
    require_image_allowed(finals, sexual=cond.sexual)
    for i, path, label in save_targets(args, plan.num_gen, metric, best):
        save_rgb(finals[i], path, sixteen_bit=bool(getattr(args, "save_16bit", False)), seed=int(args.seed) + i)
        print(f"{label}: {path}")
    after_save(args, plan, metric, best, finals, report)
    return m


__all__ = ["conditioning_image_paths", "explicit_flags", "keep_explicit", "main"]
