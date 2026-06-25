"""
Run canonical tool scripts without duplicate flat wrappers::

    python -m scripts.tools ckpt_info results/run/best.pt
    python -m scripts.tools data_quality --help
    python -m scripts.tools update_project_structure --max-depth 4

Commands accept underscores or hyphens (e.g. ``ckpt-info``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "native" / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

_HERE = Path(__file__).resolve().parent

# command name -> path to canonical script (under scripts/tools/)
_CANONICAL: dict[str, Path] = {
    "ckpt_info": _HERE / "dev" / "ckpt_info.py",
    "smoke_imports": _HERE / "dev" / "smoke_imports.py",
    "quick_test": _HERE / "dev" / "quick_test.py",
    "ar_mask_inspect": _HERE / "dev" / "ar_mask_inspect.py",
    "noise_schedule_export": _HERE / "training" / "noise_schedule_export.py",
    "mine_preference_pairs": _HERE / "training" / "mine_preference_pairs.py",
    "data_quality": _HERE / "data" / "data_quality.py",
    "manifest_paths": _HERE / "data" / "manifest_paths.py",
    "jsonl_merge": _HERE / "data" / "jsonl_merge.py",
    "manifest_gate": _HERE / "data" / "manifest_gate.py",
    "manifest_enrich": _HERE / "data" / "manifest_enrich.py",
    "caption_hygiene": _HERE / "data" / "caption_hygiene.py",
    "ar_tag_manifest": _HERE / "data" / "ar_tag_manifest.py",
    "prompt_lint": _HERE / "prompt" / "prompt_lint.py",
    "tag_coverage": _HERE / "prompt" / "tag_coverage.py",
    "suggest_style_packs": _HERE / "prompt" / "suggest_style_packs.py",
    "export_onnx": _HERE / "export" / "export_onnx.py",
    "export_safetensors": _HERE / "export" / "export_safetensors.py",
    "op_preflight": _HERE / "ops" / "op_preflight.py",
    "orchestrate_pipeline": _HERE / "ops" / "orchestrate_pipeline.py",
    "superior_generate": _HERE / "ops" / "superior_generate.py",
    "superior_dpo_loop": _HERE / "ops" / "superior_dpo_loop.py",
    "superior_curate": _HERE / "ops" / "superior_curate.py",
    "model_soup": _HERE / "ops" / "model_soup.py",
    "superior_eval_report": _HERE / "ops" / "superior_eval_report.py",
    "superior_ensemble": _HERE / "ops" / "superior_ensemble.py",
    "run_flywheel": _HERE / "ops" / "run_flywheel.py",
    "agentic_generate": _HERE / "ops" / "agentic_generate.py",
    "agentic_evolve": _HERE / "ops" / "agentic_evolve.py",
    "agentic_flywheel": _HERE / "ops" / "agentic_flywheel.py",
    "agentic_roles": _HERE / "ops" / "agentic_roles.py",
    "visual_brain": _HERE / "ops" / "visual_brain_generate.py",
    "auto_improve_loop": _HERE / "ops" / "auto_improve_loop.py",
    "gen_searcher_bridge": _HERE / "ops" / "gen_searcher_bridge.py",
    "pretrained_status": _HERE / "ops" / "pretrained_status.py",
    "startup_readiness": _HERE / "ops" / "startup_readiness.py",
    "hybrid_dit_vit_generate": _HERE / "ops" / "hybrid_dit_vit_generate.py",
    "benchmark_suite": _HERE / "benchmark_suite.py",
    "benchmark_history": _HERE / "benchmark_history.py",
    "taste_profile": _HERE / "taste_profile.py",
    "prompt_diff": _HERE / "prompt_diff.py",
    "style_gallery": _HERE / "style_gallery.py",
    "character_session": _HERE / "character_session.py",
    "auto_refine": _HERE / "auto_refine.py",
    "creative_explore": _HERE / "creative_explore.py",
    "video_generate": _HERE / "video_generate.py",
    "normalize_captions": _HERE / "normalize_captions.py",
    "preview_generation_prompt": _HERE / "preview_generation_prompt.py",
    "preview_prompt_stack": _HERE / "preview_prompt_stack.py",
    "explore_styles": _HERE / "explore_styles.py",
    "test_style_native_stack": _HERE / "dev" / "test_style_native_stack.py",
    "vit_inspect": _HERE / "vit_inspect.py",
    "seed_explorer": _HERE / "seed_explorer.py",
    "eval_prompts": _HERE / "eval_prompts.py",
    "training_timestep_preview": _HERE / "training_timestep_preview.py",
    "image_quality_qc": _HERE / "image_quality_qc.py",
    "dit_variant_compare": _HERE / "dit_variant_compare.py",
    "make_smoke_dataset": _HERE / "make_smoke_dataset.py",
    "spatial_coverage": _HERE / "spatial_coverage.py",
    "complex_prompt_coverage": _HERE / "complex_prompt_coverage.py",
    "prompt_gap_scout": _HERE / "prompt_gap_scout.py",
    "book_scene_split": _HERE / "book_scene_split.py",
    "edit_inpaint": _HERE / "edit_inpaint.py",
    "visual_memory_patch": _HERE / "visual_memory_patch.py",
    "book_prompt_audit": _HERE / "book_prompt_audit.py",
    "book_manifest_check": _HERE / "book_manifest_check.py",
    "fetch_danbooru_tags": _HERE / "fetch_danbooru_tags.py",
    "download_all_danbooru_categorized_tags": _HERE / "download_all_danbooru_categorized_tags.py",
    "split_danbooru_general_tags": _HERE / "split_danbooru_general_tags.py",
    "merge_danbooru_categorized_tags": _HERE / "merge_danbooru_categorized_tags.py",
    "train_diffusion_dpo": _HERE / "training" / "train_diffusion_dpo.py",
    "train_kd_distill": _HERE / "training" / "train_kd_distill.py",
    "train_consistency_distill": _HERE / "training" / "train_consistency_distill.py",
    "train_flow_grpo": _HERE / "training" / "train_flow_grpo.py",
    "train_ladd_distill": _HERE / "training" / "train_ladd_distill.py",
    "superior_auto_loop": _HERE / "ops" / "superior_auto_loop.py",
    "make_gallery": _HERE / "dev" / "make_gallery.py",
    "generate_sdx_architecture_diagram": _HERE / "dev" / "generate_sdx_architecture_diagram.py",
    "architecture_themes": _HERE / "dev" / "architecture_themes.py",
    "validate_config_json": _HERE / "dev" / "validate_config_json.py",
    "update_project_structure": _HERE / "repo" / "update_project_structure.py",
    "verify_doc_links": _HERE / "repo" / "verify_doc_links.py",
    "clean_repo_artifacts": _HERE / "repo" / "clean_repo_artifacts.py",
}


def _commands_help() -> str:
    names = sorted(_CANONICAL.keys())
    return "  " + "\n  ".join(names)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__ or "")
        print("Commands:\n" + _commands_help())
        return 0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help") else 1

    raw = sys.argv[1]
    cmd = raw.replace("-", "_")
    target = _CANONICAL.get(cmd)
    if target is None or not target.is_file():
        print(f"Unknown command: {raw!r}\n\n" + _commands_help(), file=sys.stderr)
        return 2

    # Delegate: script sees argv[0] as this path; remaining args unchanged.
    sys.argv = [str(target.resolve())] + sys.argv[2:]
    runpy.run_path(str(target.resolve()), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
