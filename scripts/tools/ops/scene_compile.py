#!/usr/bin/env python3
"""Compile multi-character cast / Ideogram layout / edit skills to sample argv."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cast / Ideogram / edit skill planners")
    p.add_argument("--cast", default="", help="Multi-character cast JSON")
    p.add_argument("--prompt", default="")
    p.add_argument("--ideogram", action="store_true", help="Compile Ideogram-style text/layout")
    p.add_argument("--glyph-text", action="append", default=[])
    p.add_argument("--palette", default="", help="Comma-separated hex colors")
    p.add_argument("--edit-image", default="", help="Init image for photoshop-like edit plan")
    p.add_argument("--fix-region", action="append", default=[])
    p.add_argument("--change-outfit", default="")
    p.add_argument("--work-dir", default="outputs/scene_compile")
    p.add_argument("--print-argv", action="store_true")
    p.add_argument("--json-out", default="")
    args = p.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    if args.cast:
        from utils.generation.multi_char_cast import compile_cast_scene, load_cast_scene

        scene = load_cast_scene(args.cast)
        compiled = compile_cast_scene(scene, base_prompt=args.prompt)
        out["cast"] = compiled.to_dict()
        box_path = work / "cast_box_layout.json"
        if compiled.box_layout:
            box_path.write_text(json.dumps(compiled.box_layout, indent=2), encoding="utf-8")
            out["box_layout_path"] = str(box_path)
        argv_list = ["--prompt", compiled.positive]
        if compiled.negative:
            argv_list.extend(["--negative-prompt", compiled.negative])
        if compiled.box_layout:
            argv_list.extend(["--box-layout", str(box_path), "--anti-bleed"])
        if compiled.sheet_paths:
            argv_list.extend(["--character-sheet", ",".join(compiled.sheet_paths)])
            argv_list.append("--label-multi-character-sheets")
        for fr in compiled.face_refs[:1]:
            argv_list.extend(["--reference-image", fr, "--reference-style-mode", "instantstyle"])
        out["sample_argv"] = argv_list
        print(f"cast: {len(scene.members)} characters")
        print(f"prompt: {compiled.positive[:200]}...")

    if args.ideogram or args.glyph_text:
        from utils.generation.edit_skills import plan_ideogram_layout

        palette = [c.strip() for c in args.palette.split(",") if c.strip()]
        ideo = plan_ideogram_layout(
            args.prompt or (out.get("cast") or {}).get("positive", ""),
            texts=list(args.glyph_text or None),
            palette_hex=palette,
        )
        brief_path = work / "design_brief.json"
        brief_path.write_text(json.dumps(ideo.design_brief, indent=2), encoding="utf-8")
        if ideo.box_layout:
            (work / "glyph_box_layout.json").write_text(json.dumps(ideo.box_layout, indent=2), encoding="utf-8")
        out["ideogram"] = ideo.to_dict()
        out["design_brief_path"] = str(brief_path)
        base_argv = list(out.get("sample_argv") or ideo.sample_argv)
        if "--design-brief" not in base_argv:
            base_argv.extend(["--design-brief", str(brief_path)])
        out["sample_argv"] = base_argv

    if args.edit_image:
        from utils.generation.edit_skills import plan_edit_skills

        eplan = plan_edit_skills(
            init_image=args.edit_image,
            prompt=args.prompt,
            fix_regions=list(args.fix_region or []),
            change_outfit=args.change_outfit,
            work_dir=work / "edit",
        )
        out["edit"] = eplan.to_dict()
        if eplan.sample_argv:
            out["sample_argv"] = eplan.sample_argv

    json_path = Path(args.json_out or (work / "compile.json"))
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"plan -> {json_path}")
    if args.print_argv and out.get("sample_argv"):
        print("sample argv:")
        print(" ", " ".join(shlex.quote(a) for a in out["sample_argv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
