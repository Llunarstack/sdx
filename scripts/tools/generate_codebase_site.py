#!/usr/bin/env python3
"""
Generate ``website/files.json`` and ``website/data/files-inline.js`` for the SDX codebase browser.

Includes multi-line descriptions (docstrings), and **intra-repo import** edges (imports / imported_by).
Excludes: ``docs/``, ``external/``, ``model/``, caches, etc.

Re-run: ``python scripts/tools/generate_codebase_site.py``
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from atlas_pipeline_meta import enrich_atlas_entry

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "website" / "files.json"
OUT_INLINE = ROOT / "website" / "data" / "files-inline.js"

SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cursor",
        ".vscode",
        "node_modules",
        "external",
        "docs",
        "model",
    }
)

SKIP_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".toml",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".rs",
        ".zig",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".mjs",
        ".js",
        ".ts",
        ".sh",
        ".ps1",
        ".bat",
        ".cmake",
    }
)

ALLOWED_NAMES = frozenset(
    {
        "LICENSE",
        "Makefile",
        "Dockerfile",
        ".editorconfig",
        ".gitignore",
        ".env.example",
        "CMakeLists.txt",
        "Cargo.toml",
        "Cargo.lock",
        "build.zig",
    }
)

DETAIL_MAX = 1800
IMPORT_CAP = 24
ROLE_ENTRYPOINTS = frozenset({"train.py", "sample.py", "inference.py", "scripts/cli.py"})


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _first_line(text: str, max_len: int = 240) -> str:
    text = text.strip()
    if not text:
        return ""
    line = text.split("\n")[0].strip()
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"\s+", " ", line)
    return _truncate(line, max_len)


def _md_paragraphs(raw: str, max_chars: int) -> str:
    """First few paragraphs of markdown / plain text."""
    raw = raw.strip()
    if not raw:
        return ""
    chunks: list[str] = []
    total = 0
    for block in re.split(r"\n\s*\n", raw):
        block = re.sub(r"\s+", " ", block.strip())
        if block.startswith("#"):
            block = re.sub(r"^#+\s*", "", block)
        if not block:
            continue
        if total + len(block) > max_chars:
            rest = max_chars - total
            if rest > 80:
                chunks.append(block[: rest - 1] + "…")
            break
        chunks.append(block)
        total += len(block) + 2
    return "\n\n".join(chunks)


def py_path_to_module(rel_posix: str) -> str:
    p = Path(rel_posix)
    if p.suffix != ".py":
        return ""
    stem = p.stem
    parent = p.parent
    if stem == "__init__":
        parts = parent.parts
    else:
        parts = parent.parts + (stem,)
    return ".".join(parts) if parts else stem


def build_mod_to_path(py_files: list[str]) -> dict[str, str]:
    m: dict[str, str] = {}
    for rel in py_files:
        mod = py_path_to_module(rel)
        if mod:
            m[mod] = rel
    return m


def resolve_relative_import(current_mod: str, level: int, module: str | None) -> str | None:
    parts = current_mod.split(".")
    if not parts:
        return None
    cur_pkg_parts = parts[:-1]
    up = level - 1
    if len(cur_pkg_parts) < up:
        return None
    base_parts = cur_pkg_parts[:-up] if up else cur_pkg_parts
    base = ".".join(base_parts)
    if module:
        if base:
            return f"{base}.{module}"
        return module
    return base or None


def resolve_import_to_paths(
    mod: str,
    mod_to_path: dict[str, str],
) -> list[str]:
    """Return repo-relative paths this module string might refer to."""
    if not mod:
        return []
    if mod in mod_to_path:
        return [mod_to_path[mod]]
    # subpackage import: import pkg -> pkg/__init__.py
    parts = mod.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in mod_to_path:
            return [mod_to_path[prefix]]
    return []


def extract_imports(
    rel_path: str,
    src: str,
    current_mod: str,
    mod_to_path: dict[str, str],
) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                for path in resolve_import_to_paths(name, mod_to_path):
                    if path not in seen and path != rel_path:
                        seen.add(path)
                        out.append(path)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level == 0 and node.module:
                for path in resolve_import_to_paths(node.module, mod_to_path):
                    if path not in seen and path != rel_path:
                        seen.add(path)
                        out.append(path)
            elif level > 0:
                resolved = resolve_relative_import(current_mod, level, node.module)
                if resolved:
                    for path in resolve_import_to_paths(resolved, mod_to_path):
                        if path not in seen and path != rel_path:
                            seen.add(path)
                            out.append(path)

    return out[:IMPORT_CAP]


def _py_summaries(path: Path) -> tuple[str, str]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _first_line(src), _truncate(src.strip(), DETAIL_MAX)
    doc = ast.get_docstring(tree, clean=True)
    if doc:
        summary = _first_line(doc)
        detail = _truncate(doc.strip(), DETAIL_MAX)
        return summary, detail
    return "", _truncate(src[:DETAIL_MAX], DETAIL_MAX)


def _text_summaries(path: Path) -> tuple[str, str]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", ""
    if path.suffix.lower() == ".md":
        s = _first_line(raw)
        d = _md_paragraphs(raw, DETAIL_MAX) or _truncate(raw, DETAIL_MAX)
        return s or "Markdown document.", d
    fl = _first_line(raw)
    return fl, _truncate(raw.strip(), DETAIL_MAX) or fl


def _fallback_desc(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    parent = path.parent.name or "repo root"
    if ext == ".py":
        return f"Python module in {parent}/.", f"Python source under `{parent}/`. No module docstring was parsed."
    if ext in {".rs", ".toml"}:
        return f"Rust / native tooling ({parent}/).", ""
    if ext == ".zig":
        return "Zig native helper.", ""
    if ext in {".cpp", ".h", ".hpp", ".c"}:
        return f"C/C++ source ({parent}/).", ""
    if ext == ".go":
        return "Go native helper.", ""
    if ext in {".mjs", ".js"}:
        return "JavaScript module.", ""
    if ext in {".sh", ".ps1", ".bat"}:
        return "Shell script.", ""
    if path.name in ALLOWED_NAMES:
        return f"Project file: {path.name}.", ""
    return f"Source or config ({ext or 'no ext'}).", ""


def _should_include_file(path: Path) -> bool:
    if path.name in SKIP_FILE_NAMES:
        return False
    if path.name.startswith(".") and path.name not in ALLOWED_NAMES and path.suffix == "":
        if path.name not in {".editorconfig", ".gitignore", ".env.example"}:
            return False
    name = path.name
    suf = path.suffix.lower()
    if name in ALLOWED_NAMES:
        return True
    if suf in TEXT_SUFFIXES:
        return True
    return False


def _iter_files() -> list[Path]:
    out: list[Path] = []
    root_str = str(ROOT)
    for dirpath, dirnames, filenames in os.walk(root_str):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIR_NAMES]
        rel_root = Path(dirpath).relative_to(ROOT)
        if any(p in SKIP_DIR_NAMES for p in rel_root.parts):
            continue
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if not _should_include_file(p):
                continue
            out.append(p)
    return sorted(out, key=lambda x: str(x.as_posix()).lower())


def _file_role(rel: str) -> str:
    if rel.startswith("tests/") or rel.startswith("test_"):
        return "test"
    name = Path(rel).name
    if name in ROLE_ENTRYPOINTS or rel.endswith("/__main__.py"):
        return "entrypoint"
    if rel.startswith("scripts/tools/"):
        return "tool"
    if rel.startswith("ViT/"):
        return "vit"
    if rel.startswith("models/"):
        return "model"
    if rel.startswith("diffusion/"):
        return "diffusion"
    if rel.startswith("config/"):
        return "config"
    if rel.startswith("data/"):
        return "data"
    if rel.startswith("utils/"):
        return "util"
    if rel.startswith("native/"):
        return "native"
    if rel.startswith("pipelines/"):
        return "pipeline"
    return "source"


def main() -> int:
    paths = _iter_files()
    py_rels = [p.relative_to(ROOT).as_posix() for p in paths if p.suffix == ".py"]
    mod_to_path = build_mod_to_path(py_rels)

    # First pass: summaries
    raw_entries: list[dict[str, object]] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        summary, detail = "", ""
        if path.suffix.lower() == ".py":
            summary, detail = _py_summaries(path)
        elif path.suffix.lower() in {".md", ".txt"} or path.name in {"LICENSE", ".env.example", ".editorconfig"}:
            summary, detail = _text_summaries(path)
        if not summary:
            summary, detail = _fallback_desc(path)
            if not detail:
                detail = summary
        top = rel.split("/")[0] if "/" in rel else "(root)"
        raw_entries.append(
            {
                "path": rel,
                "top": top,
                "ext": path.suffix.lower() or "(none)",
                "summary": summary,
                "detail": detail or summary,
                "role": _file_role(rel),
            }
        )

    # Second pass: imports (python only)
    path_to_imports: dict[str, list[str]] = {}
    for path in paths:
        if path.suffix != ".py":
            continue
        rel = path.relative_to(ROOT).as_posix()
        current_mod = py_path_to_module(rel)
        if not current_mod:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        path_to_imports[rel] = extract_imports(rel, src, current_mod, mod_to_path)

    imported_by: dict[str, list[str]] = {e["path"]: [] for e in raw_entries}
    for rel, targets in path_to_imports.items():
        for t in targets:
            if t in imported_by:
                imported_by[t].append(rel)

    for rel in imported_by:
        imported_by[rel] = sorted(set(imported_by[rel]))[:IMPORT_CAP]

    files_out: list[dict[str, object]] = []
    for e in raw_entries:
        rel = str(e["path"])
        d = dict(e)
        if rel in path_to_imports:
            d["imports"] = path_to_imports[rel]
        else:
            d["imports"] = []
        d["imported_by"] = imported_by.get(rel, [])
        atlas = enrich_atlas_entry(rel, str(d.get("summary", "")), str(d.get("role", "")))
        d["atlas_tags"] = atlas["atlas_tags"]
        d["atlas_summary"] = atlas["atlas_summary"]
        files_out.append(d)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": "sdx",
        "file_count": len(files_out),
        "files": files_out,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    OUT_INLINE.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, ensure_ascii=False)
    # Escape '<' so a string cannot close </script> inside embedded HTML.
    safe = json_text.replace("<", "\\u003c")
    inline_body = (
        "/* Auto-generated by scripts/tools/generate_codebase_site.py — do not edit */\n"
        f"window.__SDX_CODEBASE__ = {safe};\n"
    )
    OUT_INLINE.write_text(inline_body, encoding="utf-8")

    print(f"Wrote {len(files_out)} files -> {OUT_JSON.relative_to(ROOT)}", file=sys.stderr)
    print(f"Inline bundle -> {OUT_INLINE.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
