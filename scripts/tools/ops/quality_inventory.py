#!/usr/bin/env python3
"""
Inventory SDX quality levers: what's wired, partial, or missing.

    python -m scripts.tools quality_inventory
    python -m scripts.tools quality_inventory --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ops/ → tools/ → scripts/ → repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _collect() -> list[dict]:
    rows: list[dict] = []

    def add(name: str, status: str, path: str, note: str = "") -> None:
        rows.append({"name": name, "status": status, "path": path, "note": note})

    # Probe imports / existence
    checks = [
        ("quality_policy", "config/defaults/quality_policy.py", "compete-mode soft defaults"),
        ("layout_attn_context", "models/layout_attn_context.py", "Dense Diffusion thread state"),
        ("dense_diffusion", "frontier/attention/dense_diffusion.py", "bias_cross_attention"),
        ("anti_ai_naturalness", "models/anti_ai_naturalness.py", "FiLM naturalness"),
        ("anatomy_attention", "models/anatomy_attention.py", "spatial prior"),
        ("glyph_encoder", "utils/superior/glyph_encoder.py", "byte-hash glyphs"),
        ("glyph_projector", "utils/generation/inference_research_hooks.py", "GlyphToCondProjector"),
        ("hand_verifier", "utils/quality/hand_verifier.py", "pick-best hand heuristic"),
        ("count_binder_eval", "utils/quality/count_binder_eval.py", "count/bind prompt report"),
        ("editing_phase", "utils/generation/editing_phase.py", "post-gen edit loop"),
        ("dpo_pipeline", "utils/superior/dpo_pipeline.py", "preference alignment"),
        ("flow_grpo", "utils/training/flow_grpo.py", "online RL hooks"),
        ("reference_token_projection", "models/reference_token_projection.py", "IP-Adapter tokens"),
    ]
    for name, rel, note in checks:
        p = ROOT / rel
        status = "present" if p.is_file() else "missing"
        add(name, status, rel, note)

    # Runtime capability probes
    try:
        from models.layout_attn_context import clear_layout_attn, set_layout_attn

        clear_layout_attn()
        set_layout_attn(plan=None)
        add("layout_attn_api", "wired", "models/layout_attn_context.py", "set/clear OK")
    except Exception as e:
        add("layout_attn_api", "broken", "models/layout_attn_context.py", str(e)[:120])

    try:
        import torch
        from utils.generation.quality_stack import get_persistent_glyph_projector

        p = get_persistent_glyph_projector(32, 64, device=torch.device("cpu"), dtype=torch.float32)
        add("persistent_glyph", "wired", "utils/generation/quality_stack.py", f"proj={type(p).__name__}")
    except Exception as e:
        add("persistent_glyph", "broken", "utils/generation/quality_stack.py", str(e)[:120])

    try:
        from config.defaults.quality_policy import POLICY

        add(
            "compete_mode",
            "wired" if POLICY.aggression == "max" else "partial",
            "config/defaults/quality_policy.py",
            f"aggression={POLICY.aggression}",
        )
    except Exception as e:
        add("compete_mode", "broken", "config/defaults/quality_policy.py", str(e)[:120])

    # Gaps called out by research
    try:
        from models.dit_text_variants import CrossAttentionQKNorm
        from models.layout_attn_context import get_layout_attn  # noqa: F401

        src = CrossAttentionQKNorm.forward.__code__.co_names
        has_layout = "get_layout_attn" in src or "bias_cross_attention" in (
            CrossAttentionQKNorm.forward.__code__.co_names
        )
        # co_names may not include nested imports; check source instead
        import inspect

        body = inspect.getsource(CrossAttentionQKNorm.forward)
        has_layout = "get_layout_attn" in body and "bias_cross_attention" in body
        add(
            "dense_diffusion_in_supreme",
            "wired" if has_layout else "gap",
            "models/dit_text_variants.py CrossAttentionQKNorm",
            "layout bias on Supreme path" if has_layout else "Supreme QKNorm missing layout bias",
        )
    except Exception as e:
        add("dense_diffusion_in_supreme", "broken", "models/dit_text_variants.py", str(e)[:120])

    add(
        "dense_diffusion_in_dit",
        "wired",
        "models/dit_text.py CrossAttention",
        "reads layout_attn_context during sample_loop",
    )
    add(
        "frontier_consume",
        "present" if (ROOT / "utils/generation/frontier_consume.py").is_file() else "missing",
        "utils/generation/frontier_consume.py",
        "token emphasis + step_emphasis→noise_scales",
    )
    add(
        "repa_default",
        "partial",
        "config/train_config.py",
        "REPA coded but repa_weight defaults to 0 — use train recipe repa_weight=0.5",
    )
    add(
        "reference_adapter_train",
        "present" if (ROOT / "scripts/tools/training/train_reference_adapter.py").is_file() else "partial",
        "scripts/tools/training/train_reference_adapter.py",
        "train projector then pass --reference-adapter-pt",
    )
    add(
        "byt5_glyph_train",
        "gap",
        "utils/superior/glyph_encoder.py",
        "persistent projector ready; full ByT5 sidecar still future work",
    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Inventory SDX quality levers")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()
    rows = _collect()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'NAME':28} {'STATUS':10} PATH / NOTE")
    print("-" * 88)
    for r in rows:
        note = f" — {r['note']}" if r.get("note") else ""
        print(f"{r['name']:28} {r['status']:10} {r['path']}{note}")
    wired = sum(1 for r in rows if r["status"] in ("wired", "present"))
    gaps = sum(1 for r in rows if r["status"] in ("gap", "missing", "broken", "partial"))
    print("-" * 88)
    print(f"wired/present={wired}  partial/gap/broken={gaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
