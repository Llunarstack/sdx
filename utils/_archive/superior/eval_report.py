"""
Generate **Markdown eval reports** from ``benchmark_suite`` outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_scaffold_summary() -> List[str]:
    """Lines for eval report: local config scaffolds vs hub fallback."""
    try:
        from utils.modeling.hf_scaffold import has_local_weights, scaffold_registry
        from utils.modeling.model_paths import model_dir

        local_cfg: List[str] = []
        local_w: List[str] = []
        for e in scaffold_registry():
            fp = model_dir() / e.name
            if not fp.is_dir() or not any(fp.iterdir()):
                continue
            if has_local_weights(fp):
                local_w.append(e.name)
            else:
                local_cfg.append(e.name)
        lines: List[str] = []
        if local_w:
            suffix = " …" if len(local_w) > 12 else ""
            lines.append(f"- Weights present ({len(local_w)}): {', '.join(local_w[:12])}{suffix}")
        if local_cfg:
            suffix = " …" if len(local_cfg) > 12 else ""
            lines.append(f"- Config-only scaffolds ({len(local_cfg)}): {', '.join(local_cfg[:12])}{suffix}")
        return lines
    except Exception:
        return []


def build_markdown_report(bench_dir: Union[str, Path]) -> str:
    """Build a human-readable report from ``results.json`` + ``leaderboard.json``."""
    d = Path(bench_dir)
    results = _load_json(d / "results.json") or []
    leaderboard = _load_json(d / "leaderboard.json") or []

    lines: List[str] = [
        "# SDX Benchmark Report",
        "",
        f"Directory: `{d}`",
        "",
        "## Leaderboard",
        "",
        "| Model | Mean composite | Std | Robust score | Cases |",
        "|-------|----------------|-----|--------------|-------|",
    ]
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('model', '')} | {float(row.get('mean_composite', 0)):.3f} | "
            f"{float(row.get('std_composite', 0)):.3f} | {float(row.get('robust_score', 0)):.3f} | "
            f"{int(row.get('cases', 0))} |"
        )

    # Per-case worst scores
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        case = str(r.get("case", "") or "unknown")
        by_case.setdefault(case, []).append(r)

    lines.extend(["", "## Weakest cases (lowest composite)", ""])
    case_avgs: List[tuple[str, float]] = []
    for case, rows in by_case.items():
        vals = [float(x.get("composite", 0)) for x in rows]
        if vals:
            case_avgs.append((case, sum(vals) / len(vals)))
    case_avgs.sort(key=lambda x: x[1])
    for case, avg in case_avgs[:8]:
        lines.append(f"- **{case}**: avg composite {avg:.3f}")

    if results:
        lines.extend(["", "## Hard-negative hint", ""])
        try:
            from utils.superior.hard_negative import load_hard_negatives_from_results

            bundle = load_hard_negatives_from_results(d / "results.json")
            if bundle.negative_suffix:
                lines.append(f"Suggested extra negatives: `{bundle.negative_suffix[:200]}`")
        except Exception:
            pass

    hf_lines = _hf_scaffold_summary()
    if hf_lines:
        lines.extend(["", "## HF pretrained status", ""])
        lines.extend(hf_lines)

    try:
        from utils.modeling.text_encoder_stack import stack_download_hint, stack_status, stack_status_lines

        st = stack_status("penta")
        lines.extend(["", "## Text encoder stack (penta)", ""])
        lines.extend(stack_status_lines("penta"))
        if st.weights_count < len(st.slots):
            lines.append(f"- Download: `{stack_download_hint('penta')}`")
    except Exception:
        pass

    if results:
        modes = {
            str(r.get("text_encoder_mode", "")) for r in results if isinstance(r, dict) and r.get("text_encoder_mode")
        }
        modes.discard("")
        if modes:
            lines.extend(["", "## Checkpoint text encoder modes (from benchmark)", ""])
            for m in sorted(modes):
                try:
                    from utils.modeling.ckpt_text_stack import text_encoder_mode_label

                    lines.append(f"- **{m}**: {text_encoder_mode_label(m)}")
                except Exception:
                    lines.append(f"- **{m}**")

    try:
        from utils.modeling.hf_index import summary as hf_summary

        s = hf_summary()
        lines.extend(
            [
                "",
                "## HF registry summary",
                "",
                f"- Total registry entries: {s['total_registry']}",
                f"- Local folders: {s['local_folders']} (weights: {s['with_weights']}, config-only: {s['config_only']})",
            ]
        )
        role_counts = s.get("role_counts", {})
        if isinstance(role_counts, dict) and role_counts:
            top = ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items()))
            lines.append(f"- By role: {top}")
    except Exception:
        pass

    try:
        from utils.modeling.hf_control import controlnet_has_weights, list_controlnet_types

        ctrl_ready = [t for t in list_controlnet_types() if controlnet_has_weights(t)]
        if ctrl_ready:
            lines.extend(["", "## ControlNet weights available", ""])
            lines.append(f"- {', '.join(ctrl_ready)}")
    except Exception:
        pass

    try:
        from utils.modeling.hf_upscale import (
            face_restore_has_weights,
            list_face_restore_models,
            list_upscale_models,
            upscale_has_weights,
        )

        up = [n for n in list_upscale_models() if upscale_has_weights(n)]
        face = [n for n in list_face_restore_models() if face_restore_has_weights(n)]
        if up or face:
            lines.extend(["", "## Upscale / face-restore weights", ""])
            if up:
                lines.append(f"- Upscale: {', '.join(up)}")
            if face:
                lines.append(f"- Face restore: {', '.join(face)}")
    except Exception:
        pass

    # Optional per-result HF reward columns
    hf_keys = ("hf_reward", "hpsv2", "pickscore", "clip_h14")
    if results and any(isinstance(r, dict) and any(k in r for k in hf_keys) for r in results):
        lines.extend(["", "## HF reward averages (when logged in results.json)", ""])
        for key in hf_keys:
            vals = [float(r[key]) for r in results if isinstance(r, dict) and r.get(key) is not None]
            if vals:
                lines.append(f"- **{key}**: mean {sum(vals) / len(vals):.3f} (n={len(vals)})")

    lines.append("")
    return "\n".join(lines)


def write_report(bench_dir: Union[str, Path], out_path: Union[str, Path]) -> Path:
    md = build_markdown_report(bench_dir)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(md, encoding="utf-8")
    return op


__all__ = ["build_markdown_report", "write_report"]
