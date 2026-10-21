"""
Unified native fast paths for Style Genome (Rust / CUDA / Go).

Falls back to pure Python when optional builds are missing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .tags import token_set


def style_native_status() -> dict[str, Any]:
    """Style-genome native backend availability (``explore_styles --show-native``)."""
    out: dict[str, Any] = {"rust_style_ops": False, "cuda_style_pick": False, "go_explore": False}
    try:
        from sdx_native import cuda_style_pick_native, style_ops_native
        from sdx_native.native_tools import go_sdx_manifest_exe

        out["rust_style_ops"] = style_ops_native.available()
        out["cuda_style_pick"] = cuda_style_pick_native.available()
        out["go_explore"] = go_sdx_manifest_exe() is not None
    except Exception:
        pass
    return out


def text_overlap(a: str, b: str) -> float:
    from sdx_native.style_ops_native import maybe_token_jaccard

    j = maybe_token_jaccard(a, b)
    if j is not None:
        return j
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def pick_best_embedding_index(
    query: np.ndarray,
    candidates: np.ndarray,
) -> tuple[int, float]:
    from sdx_native.cuda_style_pick_native import maybe_pick_best_style_embedding

    picked = maybe_pick_best_style_embedding(query, candidates)
    if picked is not None:
        return picked
    qn = query / (np.linalg.norm(query) + 1e-8)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-8)
    scores = cn @ qn
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])


def explore_stats_python(manifest_path: Path) -> str:
    """Pure-Python explore manifest stats (same fields as Go ``explore-stats``)."""
    import json
    from collections import Counter

    by_genome: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    captions: set[str] = set()
    total = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            gid = str(obj.get("style_genome_id") or "(none)")
            by_genome[gid] += 1
            kind = str(obj.get("candidate_kind") or "base")
            by_kind[kind] += 1
            cap = str(obj.get("caption") or "")
            if cap:
                captions.add(cap)
    lines = [
        f"explore_manifest: {manifest_path}",
        f"  rows: {total}",
        f"  unique captions: {len(captions)}",
        "  by candidate_kind:",
    ]
    for k in sorted(by_kind):
        lines.append(f"    {k}: {by_kind[k]}")
    lines.append("  by style_genome_id:")
    for g in sorted(by_genome):
        lines.append(f"    {g}: {by_genome[g]}")
    return "\n".join(lines) + "\n"


def explore_dedupe_python(in_path: Path, out_path: Path, *, key: str = "style_genome_id") -> int:
    """Dedupe JSONL by *key*; returns number of rows written."""
    import json

    seen: dict[str, dict] = {}
    order: list[str] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = str(obj.get(key) or "")
            if not k or k in seen:
                continue
            seen[k] = obj
            order.append(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for k in order:
            f.write(json.dumps(seen[k], ensure_ascii=False) + "\n")
    return len(order)


def run_go_explore_stats(manifest_path: Path) -> str | None:
    try:
        from sdx_native.native_tools import go_sdx_manifest_exe

        exe = go_sdx_manifest_exe()
        if exe is not None:
            r = subprocess.run(
                [str(exe), "explore-stats", str(manifest_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode == 0:
                return r.stdout
    except Exception:
        pass
    if manifest_path.is_file():
        return explore_stats_python(manifest_path)
    return None


def run_go_explore_dedupe(
    in_path: Path,
    out_path: Path,
    *,
    key: str = "style_genome_id",
) -> bool:
    try:
        from sdx_native.native_tools import go_sdx_manifest_exe

        exe = go_sdx_manifest_exe()
        if exe is not None:
            r = subprocess.run(
                [str(exe), "explore-dedupe", "-o", str(out_path), "--key", key, str(in_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0:
                return True
    except Exception:
        pass
    if not in_path.is_file():
        return False
    try:
        explore_dedupe_python(in_path, out_path, key=key)
        return True
    except Exception:
        return False


__all__ = [
    "explore_dedupe_python",
    "explore_stats_python",
    "style_native_status",
    "pick_best_embedding_index",
    "run_go_explore_dedupe",
    "run_go_explore_stats",
    "text_overlap",
]
