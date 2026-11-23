#!/usr/bin/env python3
"""Book/comic run helpers: manifest check, prompt audit, scene split, visual-memory patch.

    python -m scripts.tools book manifest-check out/book_manifest.json --check-files
    python -m scripts.tools book prompt-audit pages.txt --visual-memory memory.json --cast Ren Kai
    python -m scripts.tools book scene-split story.md --out pages.txt
    python -m scripts.tools book memory-patch memory.json --entity hero --from-page 12 --patch-json "{}"

Old names (``book_manifest_check``, ``book_prompt_audit``, ``book_scene_split``, ``visual_memory_patch``)
still work as aliases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PROG = "python -m scripts.tools book"


def _manifest_check_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} manifest-check", description="Validate book_manifest.json structure and flag weak configs."
    )
    p.add_argument(
        "manifest",
        type=Path,
        help="Path to book_manifest.json, book.json, or project directory (--output-dir).",
    )
    p.add_argument(
        "--check-files", action="store_true", help="Verify each entry path exists under the project directory."
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 2 if there are warnings (default: warnings only print to stderr).",
    )
    args = p.parse_args(argv)

    from pipelines.book_comic.render import (
        load_book_manifest,
        manifest_project_root,
        manifest_summary_lines,
        validate_book_manifest,
    )

    path = args.manifest.expanduser()
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 2
    try:
        manifest = load_book_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load manifest: {exc}", file=sys.stderr)
        return 2
    errs, warns = validate_book_manifest(
        manifest, project_root=manifest_project_root(path), check_files=bool(args.check_files)
    )
    for line in manifest_summary_lines(manifest):
        print(line)
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in errs:
        print(f"ERROR: {e}", file=sys.stderr)
    if errs:
        return 1
    return 2 if warns and args.strict else 0


def _prompt_audit_main(argv: list[str]) -> int:
    from pipelines.book_comic.story import (
        approximate_token_estimate,
        composed_prompt_length_report,
        find_cast_mentions,
        load_visual_memory,
    )

    ap = argparse.ArgumentParser(
        prog=f"{_PROG} prompt-audit", description="Audit book page prompts (length, cast mentions)."
    )
    ap.add_argument("prompts_file", type=Path, help="Text file: one page prompt per line")
    ap.add_argument(
        "--warn-chars", type=int, default=0, help="Warn when a line exceeds this many characters (0 = off)."
    )
    ap.add_argument(
        "--warn-tokens", type=int, default=0, help="Warn when approximate whole-line tokens exceed this (0 = off)."
    )
    ap.add_argument(
        "--cast", nargs="*", default=[], help="Display names that should appear in each page line (substring match)."
    )
    ap.add_argument(
        "--visual-memory",
        type=Path,
        default=None,
        help="Optional visual memory JSON: adds memory fragment to length report per page.",
    )
    ap.add_argument(
        "--consistency-chars",
        type=int,
        default=0,
        help="Simulate extra consistency block length added on every page (for budgeting).",
    )
    args = ap.parse_args(argv)

    pf = args.prompts_file
    if not pf.is_file():
        print(f"error: not found: {pf}", file=sys.stderr)
        return 2
    lines = [ln.strip() for ln in pf.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        print("error: no non-empty lines", file=sys.stderr)
        return 2

    vm = None
    if args.visual_memory is not None:
        if not args.visual_memory.is_file():
            print(f"error: visual memory not found: {args.visual_memory}", file=sys.stderr)
            return 2
        vm = load_visual_memory(args.visual_memory)

    cast = [str(x).strip() for x in (args.cast or []) if str(x).strip()]
    fake_consistency = "x" * max(0, int(args.consistency_chars))

    rc = 0
    for i, page in enumerate(lines):
        vm_frag = vm.prompt_fragment_for_page(i) if vm is not None else ""
        rep = composed_prompt_length_report(
            user_prompt=page, consistency_block=fake_consistency, visual_memory_fragment=vm_frag
        )
        tot = rep["total_chars"]
        toks = approximate_token_estimate(page + " " + vm_frag + " " + fake_consistency)
        note = ""
        if cast:
            missing = find_cast_mentions(page, cast).missing
            if missing:
                note += f" missing_cast={missing}"
                rc = 1
        if args.warn_chars and tot > args.warn_chars:
            note += f" LONG>{args.warn_chars}"
            rc = 1
        if args.warn_tokens and toks > args.warn_tokens:
            note += f" TOKS>{args.warn_tokens}"
            rc = 1
        print(f"page{i:04d} chars={tot} approx_line_tokens={toks}{note}")

    print(f"summary: {len(lines)} page line(s)")
    return rc


_PAGE_HEADING = re.compile(r"^#{1,3}\s*page\s*\d+\s*$", re.I)
_PAGE_SEP = re.compile(r"^---+PAGE---+|^-{3,}\s*page\s*break\s*-{3,}\s*$", re.I)


def split_into_page_prompts(raw: str) -> list[str]:
    """Split on ``## Page N`` headings or ``---PAGE---`` lines; empty sections are dropped."""
    pages: list[str] = []
    buf: list[str] = []
    for ln in [*raw.splitlines(), "---PAGE---"]:
        if _PAGE_HEADING.match(ln.strip()) or _PAGE_SEP.match(ln.strip()):
            block = "\n".join(buf).strip()
            if block:
                pages.append(block)
            buf = []
        else:
            buf.append(ln)
    return pages


def _scene_split_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog=f"{_PROG} scene-split",
        description="Split script ('## Page 1' / ---PAGE--- sections) into one line per page for generate_book.",
    )
    p.add_argument("input", type=Path, help="Markdown or text file")
    p.add_argument("--out", type=Path, required=True, help="Output text file (one prompt per line)")
    args = p.parse_args(argv)

    pages = split_into_page_prompts(args.input.read_text(encoding="utf-8", errors="ignore"))
    if not pages:
        print("No pages found. Use '## Page 1' or ---PAGE--- between sections.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(" ".join(pg.split()) for pg in pages) + "\n", encoding="utf-8")
    print(f"Wrote {len(pages)} page line(s) -> {args.out}")
    return 0


def _memory_patch_main(argv: list[str]) -> int:
    from pipelines.book_comic.story import load_visual_memory

    ap = argparse.ArgumentParser(
        prog=f"{_PROG} memory-patch", description="Patch book visual_memory JSON (page-scoped overrides)."
    )
    ap.add_argument("memory_json", type=Path, help="Path to visual memory JSON (modified in place).")
    ap.add_argument("--from-page", type=int, required=True, help="0-based start page (inclusive).")
    ap.add_argument("--to-page", type=int, default=None, help="0-based end page (inclusive); default = from-page.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--entity", type=str, default="", help="Entity id to patch (see JSON entities map).")
    g.add_argument(
        "--page-extra",
        action="store_true",
        help="Append a run-level page_patches extra_prompt instead of an entity override.",
    )
    ap.add_argument(
        "--patch-json",
        type=str,
        default="",
        help="JSON object merged into the entity for that page range (with --entity). Omit if using --patch-file.",
    )
    ap.add_argument(
        "--patch-file",
        type=Path,
        default=None,
        help="Path to a JSON file (object) merged into the entity; avoids shell quoting issues.",
    )
    ap.add_argument("--prompt", type=str, default="", help="Free prompt fragment for --page-extra.")
    args = ap.parse_args(argv)

    path = args.memory_json
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    mem = load_visual_memory(path)

    if args.page_extra:
        prompt = (args.prompt or "").strip()
        if not prompt:
            print("error: --page-extra requires non-empty --prompt", file=sys.stderr)
            return 2
        mem.add_page_patch(from_page=args.from_page, to_page=args.to_page, extra_prompt=prompt)
    else:
        eid = (args.entity or "").strip()
        if not eid:
            print("error: --entity required unless --page-extra", file=sys.stderr)
            return 2
        if args.patch_file is not None and not args.patch_file.is_file():
            print(f"error: --patch-file not found: {args.patch_file}", file=sys.stderr)
            return 2
        raw = (
            args.patch_file.read_text(encoding="utf-8") if args.patch_file else (args.patch_json or "").strip() or "{}"
        )
        try:
            patch = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"error: invalid patch JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(patch, dict):
            print("error: patch must be a JSON object", file=sys.stderr)
            return 2
        mem.apply_entity_page_patch(eid, from_page=args.from_page, to_page=args.to_page, patch=patch)

    mem.save(path)
    print(f"updated {path}")
    return 0


COMMANDS = {
    "manifest-check": _manifest_check_main,
    "prompt-audit": _prompt_audit_main,
    "scene-split": _scene_split_main,
    "memory-patch": _memory_patch_main,
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
