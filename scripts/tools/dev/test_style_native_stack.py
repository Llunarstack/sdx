#!/usr/bin/env python3
"""Smoke-test Style Genome + native stack (Rust / CUDA / Go / Mojo)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "native" / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np


def main() -> int:
    failures: list[str] = []

    # 1) Discovery
    from utils.prompt.style_native import native_stack_status

    st = native_stack_status()
    print("native_stack_status:", json.dumps(st, indent=2))

    # 2) Rust style ops
    from sdx_native.style_ops_native import (
        maybe_fnv1a64,
        maybe_merge_style_axes,
        maybe_token_jaccard,
    )

    j = maybe_token_jaccard("red dress, blue", "red dress, golden")
    if j is None:
        failures.append("Rust sdx_token_jaccard_utf8 unavailable (rebuild sdx-prompt-ops --release)")
    else:
        print(f"Rust Jaccard: {j:.4f}")
        assert 0.0 < j <= 1.0

    fp = maybe_fnv1a64("glitch cathedral palette")
    if fp is None:
        failures.append("Rust sdx_fnv1a64_utf8 unavailable")
    else:
        print(f"Rust FNV: {fp:#018x}")

    merged = maybe_merge_style_axes(["a, b", "B, c", "d"])
    if merged is None:
        failures.append("Rust sdx_merge_style_axes_utf8 unavailable")
    else:
        print(f"Rust merge axes: {merged!r}")
        assert merged == "a, b, c, d"

    # 3) Inventor + chaos
    from utils.prompt.style_inventor import StyleInventor
    from utils.prompt.style_genome_chaos import invent_insane_batch, fuse_genomes

    inv = StyleInventor(use_qwen=False)
    genomes = inv.invent(
        "samurai at sunset", n=2, invention_mode="insane", chaos_level=0.8, seed=42
    )
    assert len(genomes) == 2
    print(f"Invented: {[g.name for g in genomes]}")
    ch = fuse_genomes(genomes[0], genomes[1])
    print(f"Chimera: {ch.name}")

    batch = invent_insane_batch("void", n=2, mode="apocalypse", chaos_level=1.0, seed=1)
    assert len(batch) == 2

    # 4) Explore planner + manifest
    from utils.prompt.style_explore import StyleExplorePlanner

    out = _ROOT / "data" / "style_genomes" / "_smoke_explore.jsonl"
    planner = StyleExplorePlanner(use_qwen=False)
    plan = planner.plan(
        "portrait in rain",
        n_genomes=2,
        mutations_per_genome=1,
        invention_mode="insane",
        chaos_level=0.7,
        fusion_pairs=True,
        hypermutate_siblings=1,
        seed=7,
    )
    plan.write_manifest(out)
    assert out.is_file() and out.stat().st_size > 0
    print(f"Manifest: {len(plan.candidates)} candidates -> {out}")

    # 5) PromptStack preview with genome
    from types import SimpleNamespace

    from utils.prompt.stack import PromptContext, StackMode, run_prompt_stack

    g0 = plan.candidates[0].genome
    ns = SimpleNamespace(
        prompt=plan.candidates[0].positive,
        negative_prompt=plan.candidates[0].negative,
        style=plan.candidates[0].style,
        invent_styles=0,
        style_genome_file="",
        _active_style_genome=g0,
        style_chaos_level=0.5,
        shortcomings_mitigation="none",
        art_guidance_mode="none",
        anatomy_guidance="none",
        style_guidance_mode="none",
        auto_content_fix=False,
        prompt_stack_intelligence=True,
        prompt_stack_auto_quality=True,
        prompt_clauses="style.chaos,style.surreal",
        prompt_special_helpers="off",
        one_shot_boost=False,
        text_in_image=False,
        no_neg_filter=False,
    )
    ctx = PromptContext(
        positive=plan.candidates[0].positive,
        negative=plan.candidates[0].negative,
        mode=StackMode.PREVIEW,
        args=ns,
    )
    result = run_prompt_stack(ctx)
    assert "style_genome" in ".".join(result.trace)
    print("PromptStack trace:", " -> ".join(result.trace[:8]), "...")

    # 6) CUDA pick-best (numpy fallback ok)
    from utils.prompt.style_native import pick_best_embedding_index

    dim = 16
    q = np.random.randn(dim).astype(np.float32)
    c = np.random.randn(4, dim).astype(np.float32)
    c[2] = q + 0.001
    idx, score = pick_best_embedding_index(q, c)
    print(f"Pick-best embedding: idx={idx} score={score:.4f}")
    assert idx == 2

    # 7) Go explore-stats (optional)
    from utils.prompt.style_native import run_go_explore_stats

    stats = run_go_explore_stats(out)
    if stats:
        print("Go explore-stats:\n", stats)
    else:
        print("Go explore-stats: skipped (sdx-manifest not built)")

    # 8) Mojo merge (optional)
    from sdx_native.style_tokens_mojo import merge_comma_dedupe, mojo_available

    m = merge_comma_dedupe("x, y, X, z")
    print(f"Comma dedupe: {m!r} (mojo={mojo_available()})")
    assert "x" in m and "y" in m and "z" in m

    if failures:
        print("\nWARNINGS (non-fatal):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
    else:
        print("\nAll native style stack checks passed.")
    return 0 if not failures else 0  # warnings only


if __name__ == "__main__":
    raise SystemExit(main())
