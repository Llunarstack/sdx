"""
Run any tool script under ``scripts/tools/`` by its file name::

    python -m scripts.tools ckpt_info results/run/best.pt
    python -m scripts.tools data_quality --help
    python -m scripts.tools update_project_structure --max-depth 4

Every ``scripts/tools/**/<name>.py`` with a ``__main__`` guard is a command; names accept underscores
or hyphens (e.g. ``ckpt-info``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from scripts.tools._repo_bootstrap import ensure_repo_paths

ensure_repo_paths()

from utils.runtime import configure_stdio_for_console  # noqa: E402

# Tool scripts don't all guard their own stdio; without this, argparse help
# containing arrows crashes on legacy Windows console encodings (cp1252).
configure_stdio_for_console()

_HERE = Path(__file__).resolve().parent
# alias -> "script [subcommand ...]"; merged scripts keep their old command names here.
_ALIASES = {
    "feedback": "superior feedback",
    "feedback_cli": "superior feedback",
    "auto_improve_loop": "superior auto-improve",
    "mine_preference_pairs": "superior mine-pairs",
    "preference_flywheel": "superior prefs",
    "rsi_loop": "superior rsi",
    "benchmark_suite": "bench suite",
    "benchmark_history": "bench history",
    "eval_prompts": "bench prompts",
    "seed_explorer": "bench seeds",
    "make_gallery": "bench gallery",
    "run_eval": "evals",
    "pretrained_status": "status pretrained",
    "startup_readiness": "status readiness",
    "quality_inventory": "status inventory",
    "gen_searcher_bridge": "status gen-searcher",
    "train_diffusion_dpo": "trainers dpo",
    "train_flow_grpo": "trainers flow-grpo",
    "train_consistency_distill": "trainers consistency-distill",
    "train_kd_distill": "trainers kd-distill",
    "train_ladd_distill": "trainers ladd-distill",
    "train_reference_adapter": "trainers reference-adapter",
    "train_style_lora_oneshot": "trainers style-lora-oneshot",
    "train_video": "trainers video",
    "precompute_latents": "trainers precompute-latents",
    "train_with_expandable_segments": "trainers expandable-train",
    "ckpt_info": "ckpt info",
    "vit_inspect": "ckpt vit-inspect",
    "export_onnx": "ckpt export-onnx",
    "export_safetensors": "ckpt export-safetensors",
    "model_soup": "ckpt soup",
    "noise_schedule_export": "ckpt noise-schedule",
    "ar_mask_inspect": "ckpt ar-mask",
    "dit_variant_compare": "ckpt dit-compare",
    "training_timestep_preview": "ckpt timestep-preview",
    "train_enhanced": "enhanced train",
    "sample_enhanced": "enhanced sample",
    "save_model_checkpoint": "enhanced init",
    "corpus_retrieve": "manifest corpus",
    "data_quality": "manifest quality",
    "manifest_enrich": "manifest enrich",
    "manifest_gate": "manifest gate",
    "image_quality_qc": "manifest image-qc",
    "manifest_paths": "manifest paths",
    "jsonl_merge": "manifest merge",
    "ar_tag_manifest": "manifest ar-tag",
    "make_smoke_dataset": "manifest smoke-dataset",
    "hf_export_to_sdx_manifest": "manifest hf-export",
    "hf_download_and_train": "manifest hf-train",
    "normalize_captions": "captions normalize",
    "caption_hygiene": "captions hygiene",
    "prompt_lint": "captions lint",
    "tag_coverage": "captions tag-coverage",
    "spatial_coverage": "captions spatial-coverage",
    "complex_prompt_coverage": "captions complex-coverage",
    "prompt_gap_scout": "captions gap-scout",
    "op_preflight": "captions preflight",
    "fetch_danbooru_tags": "danbooru fetch",
    "split_danbooru_general_tags": "danbooru split",
    "merge_danbooru_categorized_tags": "danbooru merge",
    "download_all_danbooru_categorized_tags": "danbooru all",
    "superior_generate": "superior generate",
    "superior_ensemble": "superior ensemble",
    "superior_curate": "superior curate",
    "superior_eval_report": "superior report",
    "superior_dpo_loop": "superior dpo",
    "superior_auto_loop": "superior auto-loop",
    "run_flywheel": "superior flywheel",
    "agentic_generate": "agentic generate",
    "agentic_roles": "agentic roles",
    "agentic_evolve": "agentic evolve",
    "agentic_flywheel": "agentic flywheel",
    "agentic_stills": "agentic stills",
    "perfect_gen": "agentic perfect",
    "visual_brain": "agentic visual-brain",
    "visual_brain_generate": "agentic visual-brain",
    "hybrid_dit_vit_generate": "generate hybrid",
    "orchestrate_pipeline": "generate orchestrate",
    "auto_refine": "generate auto-refine",
    "edit_inpaint": "generate edit-inpaint",
    "editing_phase": "generate editing-phase",
    "cascade_generate": "generate cascade",
    "profile_image_cli": "generate profile-image",
    "video_generate": "pipelines/video/scripts/generate_video.py",
    "preview_prompt_stack": "prompts preview",
    "preview_generation_prompt": "prompts preview",
    "prompt_compose": "prompts compose",
    "prompt_diff": "prompts diff",
    "research_image_prompt": "prompts research-image",
    "suggest_style_packs": "prompts suggest-styles",
    "explore_styles": "prompts explore-styles",
    "creative_explore": "prompts creative-explore",
    "style_gallery": "prompts style-gallery",
    "taste_profile": "prompts taste",
    "character_session": "prompts character-session",
    "invention_lab": "prompts invention-lab",
    "scene_compile": "prompts scene-compile",
    "creative_copilot": "prompts copilot",
    "book_manifest_check": "book manifest-check",
    "book_prompt_audit": "book prompt-audit",
    "book_scene_split": "book scene-split",
    "visual_memory_patch": "book memory-patch",
}


def commands() -> dict[str, tuple[Path, list[str]]]:
    """Command name -> (script path, leading args): file stems plus ``_ALIASES``."""
    out: dict[str, tuple[Path, list[str]]] = {
        p.stem: (p, [])
        for p in sorted(_HERE.rglob("*.py"))
        if not p.name.startswith("_")
        and "__pycache__" not in p.parts
        and 'if __name__ == "__main__"' in p.read_text(encoding="utf-8", errors="ignore")
    }
    for alias, spec in _ALIASES.items():
        target, *lead = spec.split()
        if target in out:
            out[alias] = (out[target][0], lead)
        elif target.endswith(".py"):  # repo-relative script outside scripts/tools
            out[alias] = (_HERE.parents[1] / target, lead)
    return out


def main() -> int:
    cmds = commands()
    listing = "Commands:\n  " + "\n  ".join(sorted(cmds))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__ or "")
        print(listing)
        return 0 if len(sys.argv) >= 2 else 1

    raw = sys.argv[1]
    found = cmds.get(raw.replace("-", "_"))
    if found is None:
        print(f"Unknown command: {raw!r}\n\n{listing}", file=sys.stderr)
        return 2

    # The script sees argv[0] as its own path, then any alias subcommand, then the user's args.
    target, lead = found
    sys.argv = [str(target), *lead, *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
