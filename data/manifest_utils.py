"""Shared manifest helpers for scrape -> enrich -> train and the JSONL CLI tools.

``python -m data.manifest_utils path/to/manifest.jsonl [--rust-stats] [--json-out]`` prints quick JSONL stats
(rows, keys, caption-length percentiles) without loading images.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_manifest_rows(path: Path) -> Iterator[dict]:
    """Stream JSONL rows, skipping blank and malformed lines."""
    with Path(path).open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_manifest_rows(path: Path) -> list[dict]:
    return list(iter_manifest_rows(path)) if path.is_file() else []


def first_str(row: dict, keys: Iterable[str]) -> str:
    """First non-empty string value of ``row`` among ``keys`` (stripped), else ``""``."""
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def caption_from_row(row: dict) -> str:
    return first_str(row, ("caption", "text", "prompt", "instruction"))


def resolve_image_path(rel: str, *roots: Path | None) -> Path | None:
    """``rel`` itself if it is a file, else the first ``root / rel`` that is; ``None`` when nothing exists."""
    if not rel:
        return None
    for cand in (Path(rel), *(root / rel for root in roots if root is not None)):
        if cand.is_file():
            return cand
    return None


def manifest_has_rows(path: Path) -> bool:
    """True when ``path`` is a file with at least one parseable row (stops at the first)."""
    return path.is_file() and any(True for _ in iter_manifest_rows(path))


def pick_training_manifest(combined: Path, enriched: Path) -> Path:
    """Prefer enriched manifest when it has rows (avoids empty partial writes)."""
    if manifest_has_rows(enriched):
        return enriched
    if combined.is_file():
        return combined
    raise FileNotFoundError(f"no manifest (combined={combined}, enriched={enriched})")


def negative_caption_from_row(row: dict) -> str:
    """Map enrich / research fields to the training negative caption."""
    return first_str(row, ("negative_caption", "negative_prompt", "negative_prompt_hint"))


_SCREENED_FIELDS = (
    "caption",
    "text",
    "instruction",
    "caption_global",
    "caption_local",
    "entity_captions",
    "region_captions",
    "segments",
    "parts",
    "character_tags",
    "copyright_tags",
    "wd_tags",
    "prompt_layout",
)


def _field_text(val, *, space_separated: bool = False) -> str:
    if isinstance(val, str):
        return ", ".join(val.split()) if space_separated else val
    if isinstance(val, dict):
        val = list(val.values())
    if isinstance(val, (list, tuple)):
        return ", ".join(t for t in (_field_text(v) for v in val) if t)
    return ""


def manifest_row_text(row: dict) -> str:
    """Every caption, tag and layout field of a manifest row, for training-data safety screening.

    Booru ``tag_string*`` fields (and a comma-free ``tags`` string) are space-separated tags,
    so they are re-joined with commas.
    """
    parts = [_field_text(row.get(k)) for k in _SCREENED_FIELDS]
    tags = row.get("tags")
    parts.append(_field_text(tags, space_separated=isinstance(tags, str) and "," not in tags))
    parts += [_field_text(v, space_separated=True) for k, v in row.items() if k.startswith("tag_string")]
    return ", ".join(p for p in parts if p)


def digest_jsonl(path: Path, *, max_key_samples: int = 12) -> dict[str, Any]:
    """Row / key / image / caption counts, JSON parse errors and caption-length percentiles."""
    n = parse_errors = has_image = has_caption = 0
    key_counts: Counter[str] = Counter()
    cap_lens: list[int] = []
    sample_paths: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            if not isinstance(row, dict):
                parse_errors += 1
                continue
            key_counts.update(row.keys())
            ip = first_str(row, ("image_path", "path", "image"))
            if ip:
                has_image += 1
                if len(sample_paths) < 3:
                    sample_paths.append(ip[:120])
            cap = first_str(row, ("caption", "text", "prompt"))
            if cap:
                has_caption += 1
                cap_lens.append(len(cap))
    cap_lens.sort()

    def pct(q: float) -> int:
        return cap_lens[min(len(cap_lens) - 1, int((len(cap_lens) - 1) * q + 0.5))]  # round half up, as in Rust stats

    return {
        "path": str(path.resolve()),
        "lines_read": n,
        "json_parse_errors": parse_errors,
        "rows_with_image_field": has_image,
        "rows_with_caption_like": has_caption,
        "caption_len_chars": (
            {"min": cap_lens[0], "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": cap_lens[-1]}
            if cap_lens
            else {}
        ),
        "distinct_keys_top": [k for k, _ in key_counts.most_common(max_key_samples)],
        "sample_image_paths": sample_paths,
    }


def try_rust_stats(manifest: Path) -> str | None:
    """Run ``sdx-jsonl-tools stats`` when the release binary exists (via ``native_tools``)."""
    try:
        from utils.native import run_rust_jsonl_stats, rust_jsonl_tools_exe
    except ImportError:
        return None
    if rust_jsonl_tools_exe() is None:
        return None
    try:
        r = run_rust_jsonl_stats(manifest, timeout=600)
        if r.returncode != 0:
            return r.stderr or r.stdout or "rust stats failed"
        return (r.stdout or "").strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def main() -> int:
    p = argparse.ArgumentParser(description="JSONL manifest digest (quick QoL)")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--rust-stats", action="store_true", help="Append sdx-jsonl-tools stats if built")
    p.add_argument("--json-out", action="store_true", help="Print digest as JSON only")
    args = p.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    d = digest_jsonl(path)
    if args.rust_stats:
        d["rust_stats_stdout"] = try_rust_stats(path)
    if args.json_out:
        print(json.dumps(d, indent=2))
        return 0

    print(f"=== manifest digest: {d['path']} ===")
    for key in ("lines_read", "json_parse_errors", "rows_with_image_field", "rows_with_caption_like"):
        print(f"{key}: {d[key]}")
    if d["caption_len_chars"]:
        print("caption_len_chars: " + " ".join(f"{k}={v}" for k, v in d["caption_len_chars"].items()))
    print(f"top_keys: {d['distinct_keys_top']}")
    if d["sample_image_paths"]:
        print("sample paths:", d["sample_image_paths"])
    if args.rust_stats:
        print("--- rust stats ---")
        print(d.get("rust_stats_stdout") or "(sdx-jsonl-tools not built or failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
