#!/usr/bin/env python3
"""Generate video via retrieve → keyframe edit → interpolate → stitch pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Tip: use --scene examples/scene.example.json so you only edit ONE file.",
    )
    p.add_argument(
        "--scene",
        type=str,
        default="",
        help="Scene graph JSON (recommended). Replaces --prompt and most flags.",
    )
    p.add_argument("--mode", choices=("t2v", "i2v"), default="t2v")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--out", type=str, default="runs/video/output.mp4")
    p.add_argument("--ckpt", type=str, default="", help="SDX checkpoint for keyframe img2img edits")
    p.add_argument("--work-dir", type=str, default="runs/video")
    p.add_argument("--anchor-image", type=str, default="", help="Required for i2v")
    p.add_argument("--motion-clip", type=str, default="", help="Reference clip for i2v motion")
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--local-library", type=str, default="", help="Directory of reference mp4/mov clips")
    p.add_argument("--catalog", type=str, default="data/video_catalog.json")
    p.add_argument("--use-pexels", action="store_true", help="Search Pexels when PEXELS_API_KEY set")
    p.add_argument("--allow-download", action="store_true", help="Download remote clips (requires --use-pexels)")
    p.add_argument("--keyframe-interval", type=int, default=6)
    p.add_argument("--dry-run", action="store_true", help="Skip sample.py; motion+interpolate only")
    p.add_argument("--control-help", action="store_true", help="Print control mode reference and exit")
    p.add_argument("--list-frontier", action="store_true", help="List novel frontier video modules and exit")
    p.add_argument("--plan-only", action="store_true", help="Print shot plan JSON and exit")
    p.add_argument("--validate-scene", action="store_true", help="Validate scene JSON and exit")
    p.add_argument("--preflight", action="store_true", help="Run preflight checks (paths, ckpt) and exit")
    p.add_argument("--validate-continuity", action="store_true", help="Run continuity validators and exit")
    p.add_argument("--thumbnail-plan", action="store_true", help="Print thumbnail rehearsal plan and exit")
    p.add_argument("--preview-retrieval", action="store_true", help="Print retrieval rankings and exit")
    args, extra = p.parse_known_args()

    if args.control_help:
        from pipelines.video.controls import control_mode_help

        for k, v in control_mode_help().items():
            print(f"  {k}: {v}")
        return 0

    if args.list_engines:
        from pipelines.video.style_engines import list_engines

        for e in list_engines():
            print(f"  {e.id.value}: {e.title} — {e.pipeline_notes or e.positive[:60]}")
        return 0

    if args.list_frontier:
        from pipelines.video.frontier_compiler import list_frontier_modules

        for m in list_frontier_modules():
            print(f"  {m['id']}: {m['module']} — {m['summary']}")
        return 0

    if args.scene:
        from pipelines.video.scene_graph import compile_scene_file, load_scene_graph, validate_scene_graph

        if args.preflight:
            from pipelines.video.scene_preflight import format_preflight_report, run_preflight

            report = run_preflight(args.scene, ckpt=args.ckpt, require_ckpt=bool(args.ckpt))
            print(format_preflight_report(report))
            return 0 if report.ok else 1

        graph = load_scene_graph(args.scene)
        issues = validate_scene_graph(graph)
        if args.validate_continuity:
            from pipelines.video.continuity_validators import (
                format_continuity_report,
                parse_validator_config,
                run_continuity_validation,
            )

            shots = list(graph.shots) if graph.shots else []
            report = run_continuity_validation(
                shots, continuity=graph.continuity, config=parse_validator_config(graph.continuity)
            )
            print(format_continuity_report(report))
            return 0 if report.ok else 1

        if args.thumbnail_plan:
            from dataclasses import asdict

            from pipelines.video.thumbnail_rehearsal import parse_thumbnail_config, plan_thumbnails

            thumb_cfg = parse_thumbnail_config(
                graph.continuity,
                studio=graph.raw.get("studio") if isinstance(graph.raw.get("studio"), dict) else None,
                edit=graph.edit,
            )
            shots = list(graph.shots) if graph.shots else []
            if not shots:
                from pipelines.video.scene_graph import compile_scene_graph

                try:
                    compiled = compile_scene_graph(graph)
                    shots = compiled.graph.shots or []
                except ValueError:
                    shots = []
            plan = plan_thumbnails(
                shots,
                config=thumb_cfg,
                base_prompt=graph.scene_prompt,
                aspect_width=graph.width,
                aspect_height=graph.height,
            )
            print(
                json.dumps(
                    {
                        "enabled": plan.enabled,
                        "gate_passed": plan.gate_passed,
                        "pending_count": plan.pending_count,
                        "config": {
                            "size": thumb_cfg.size,
                            "gate": thumb_cfg.gate,
                            "frames_per_shot": thumb_cfg.frames_per_shot,
                        },
                        "specs": [asdict(t) for t in plan.specs],
                    },
                    indent=2,
                )
            )
            return 0

        if args.validate_scene:
            if issues:
                print("INVALID:")
                for x in issues:
                    print(f"  - {x}")
                return 1
            print("OK:", args.scene)
            return 0
        if issues:
            print("Scene errors:", file=sys.stderr)
            for x in issues:
                print(f"  - {x}", file=sys.stderr)
            return 1
        if args.plan_only:
            compiled = compile_scene_file(args.scene)
            print(
                json.dumps(
                    {
                        "scene_prompt": compiled.plan.user_prompt,
                        "shots": [s.__dict__ for s in compiled.plan.shots],
                        "segment_overrides": compiled.segment_overrides,
                    },
                    indent=2,
                )
            )
            return 0
        from pipelines.video.pipeline import run_from_scene_file

        result = run_from_scene_file(
            args.scene,
            args.out,
            ckpt=args.ckpt,
            work_dir=args.work_dir,
            dry_run=args.dry_run,
            use_pexels=args.use_pexels,
            allow_download=args.allow_download,
            sample_extra_args=extra,
        )
        print(f"Output: {result.output_path}")
        return 0

    if not args.prompt:
        p.error("Provide --prompt or --scene")
        from pipelines.video.helpers import preview_shot_plan

        plan = preview_shot_plan(
            args.prompt,
            mode=args.mode,
            duration_sec=args.duration,
            fps=args.fps,
        )
        print(
            json.dumps(
                {
                    "mode": plan.mode.value,
                    "timeline": plan.timeline.__dict__,
                    "shots": [s.__dict__ for s in plan.shots],
                },
                indent=2,
            )
        )
        return 0

    if args.preview_retrieval:
        from pipelines.video.helpers import preview_retrieval_rankings

        print(
            json.dumps(
                preview_retrieval_rankings(args.prompt, local_library=args.local_library, catalog_path=args.catalog),
                indent=2,
            )
        )
        return 0

    from pipelines.video.pipeline import run_video_pipeline

    result = run_video_pipeline(
        mode=args.mode,
        prompt=args.prompt,
        out_path=args.out,
        anchor_image=args.anchor_image,
        ckpt=args.ckpt,
        work_dir=args.work_dir,
        motion_clip=args.motion_clip,
        duration_sec=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        local_library=args.local_library,
        catalog_path=args.catalog,
        use_pexels=args.use_pexels,
        allow_download=args.allow_download,
        dry_run=args.dry_run,
        keyframe_interval=args.keyframe_interval,
        sample_extra_args=extra,
    )
    print(f"Output: {result.output_path}")
    print(f"Segments: {len(result.segment_paths)}")
    for q in result.quality:
        status = "PASS" if q.passed else "FAIL"
        print(f"  seg {q.segment_index}: {status} temporal={q.temporal_score:.3f} sharp={q.sharpness_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
