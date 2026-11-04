#!/usr/bin/env python3
"""Environment / model status reports: pretrained, readiness, inventory, gen-searcher.

    python -m scripts.tools status pretrained [--summary | --role vlm | --text-encoder-mode penta]
    python -m scripts.tools status readiness --dataset-manifest data/train.jsonl --out-md startup.md
    python -m scripts.tools status inventory [--json]
    python -m scripts.tools status gen-searcher --in-json gs_out.jsonl --prompt "..." --out-merged-prompt p.txt

Old names still work: ``pretrained_status``, ``startup_readiness``, ``quality_inventory``, ``gen_searcher_bridge``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_json(path_s: str, payload: Any) -> Path:
    p = Path(path_s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------- pretrained


def _folder_size_gb(path_s: str) -> float:
    p = Path(path_s)
    if not p.is_dir():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total / 1024**3


def build_pretrained_report() -> dict:
    from utils.modeling.model_paths import pretrained_catalog, verify_gen_searcher_8b_local

    rows = pretrained_catalog()
    for r in rows:
        resolved = str(r["resolved"])
        is_local = Path(resolved).is_absolute() and Path(resolved).exists()
        r["is_local"] = is_local
        r["size_gb"] = round(_folder_size_gb(resolved), 3) if is_local else 0.0
        if str(r["name"]) == "GenSearcher-8B" and is_local:
            gs = verify_gen_searcher_8b_local(resolved)
            r["gen_searcher_ok"] = bool(gs.get("all_required_present", False))
            r["gen_searcher_missing"] = list(gs.get("missing", []))
    return {"models": rows}


def _pretrained_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="status pretrained", description="Show SDX pretrained model wiring status.")
    ap.add_argument("--out-json", type=str, default="", help="Optional JSON path for machine-readable report.")
    ap.add_argument(
        "--role", action="append", default=[], help="Filter by HF scaffold role (vlm, reward, control, ...)."
    )
    ap.add_argument("--summary", action="store_true", help="Print HF registry summary only.")
    ap.add_argument(
        "--text-encoder-mode",
        type=str,
        default="",
        choices=["", "t5", "triple", "penta"],
        help="Show readiness for a text encoder stack (t5 / triple / penta).",
    )
    args = ap.parse_args(argv)

    if args.text_encoder_mode:
        from utils.modeling.encoders import stack_download_hint, stack_status, stack_status_lines

        mode = str(args.text_encoder_mode)
        st = stack_status(mode)
        print(f"Text encoder stack ({mode}):")
        for line in stack_status_lines(mode):
            print(line)
        print(f"Ready for training/sampling: {st.ready}")
        if st.weights_count < len(st.slots):
            print(f"Download hint:\n  {stack_download_hint(mode)}")
        return 0

    if args.summary:
        from utils.modeling.model_paths import registry_summary, role_counts

        s = registry_summary()
        print(
            f"HF registry: total={s['total_registry']} local={s['local_folders']} "
            f"weights={s['with_weights']} config_only={s['config_only']}"
        )
        print("Roles:", ", ".join(f"{k}={v}" for k, v in role_counts().items()))
        return 0

    report = build_pretrained_report()
    if args.role:
        from utils.modeling.model_paths import scaffold_registry

        role_set = {r.strip().lower() for r in args.role}
        names = {e.name for e in scaffold_registry() if e.role.lower() in role_set}
        report["models"] = [r for r in report["models"] if str(r.get("name")) in names]
    print("SDX pretrained resolution status:")
    for r in report["models"]:
        line = f"- {r['name']}: {'local' if r.get('is_local') else 'hf'} -> {r['resolved']}"
        if r.get("is_local"):
            line += f" ({r.get('size_gb', 0.0):.3f} GB)"
        if r.get("name") == "GenSearcher-8B":
            line += f" | gen_searcher_ok={bool(r.get('gen_searcher_ok', False))}"
        print(line)
    if str(args.out_json).strip():
        p = _write_json(str(args.out_json).strip(), report)
        print(f"[pretrained_status] wrote report -> {p}")
    return 0


# ---------------------------------------------------------------- readiness (no-train launch audit)


def _package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _check_packages(required: list[str], optional: list[str]) -> dict[str, dict[str, bool]]:
    return {
        "required": {k: _package_available(k) for k in required},
        "optional": {k: _package_available(k) for k in optional},
    }


def _disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1024**3
    except Exception:
        return 0.0


def _gpu_status() -> dict[str, Any]:
    out: dict[str, Any] = {"torch_imported": False, "cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch  # type: ignore
    except Exception:
        return out
    out["torch_imported"] = True
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
        if not out["cuda_available"]:
            return out
        out["device_count"] = int(torch.cuda.device_count())
    except Exception:
        return out
    for i in range(out["device_count"]):
        row: dict[str, Any] = {"index": i, "name": "", "total_vram_gb": 0.0}
        try:
            row["name"] = str(torch.cuda.get_device_name(i))
            row["total_vram_gb"] = torch.cuda.get_device_properties(i).total_memory / 1024**3
        except Exception:
            pass
        out["devices"].append(row)
    return out


def _gather_native_status() -> dict[str, Any]:
    try:
        from utils.native import native_stack_status

        return dict(native_stack_status())
    except Exception as exc:
        return {"error": str(exc)}


def _native_presence_score(native_status: dict[str, Any]) -> tuple[int, int]:
    keys = sorted(k for k in native_status if k.startswith(("rust_", "zig_", "go_", "libsdx_")))
    have = 0
    for k in keys:
        v = native_status.get(k)
        if isinstance(v, bool):
            have += int(v)
        elif isinstance(v, str):
            have += int(bool(v.strip()))
    return have, len(keys)


def build_readiness_report(*, repo_root: Path, dataset_manifest: str = "", work_dir: str = "") -> dict[str, Any]:
    pkgs = _check_packages(["numpy", "PIL", "torch"], ["diffusers", "transformers", "cv2", "pytesseract"])
    gpu = _gpu_status()
    native = _gather_native_status()
    tools = repo_root / "scripts" / "tools"
    key_files = {
        str(p): p.exists()
        for p in (repo_root / "sample.py", repo_root / "train.py", tools / "bench.py", tools / "superior.py")
    }

    disk_paths = [repo_root] + ([Path(work_dir).resolve()] if work_dir.strip() else [])
    disk = {str(p): {"free_gb": _disk_free_gb(p)} for p in disk_paths}

    dataset_ok = True
    dataset_path = ""
    if dataset_manifest.strip():
        dataset_path = str(Path(dataset_manifest).resolve())
        dataset_ok = Path(dataset_path).is_file()

    blockers: list[str] = []
    warnings: list[str] = []
    missing = [k for k, ok in pkgs["required"].items() if not ok]
    if missing:
        blockers.append(f"Missing required packages: {', '.join(missing)}")
    missing_files = [k for k, ok in key_files.items() if not ok]
    if missing_files:
        blockers.append(f"Missing key files: {', '.join(missing_files)}")
    if not dataset_ok:
        blockers.append(f"Dataset manifest not found: {dataset_path}")
    if not gpu.get("cuda_available", False):
        warnings.append("CUDA not detected; future tr/inference will be slower on CPU.")
    low_disk = [k for k, row in disk.items() if float(row.get("free_gb", 0.0)) < 20.0]
    if low_disk:
        warnings.append(f"Low disk headroom (<20GB free): {', '.join(low_disk)}")
    native_have, native_total = _native_presence_score(native)
    if native_total > 0 and native_have == 0:
        warnings.append("No native accelerators detected; pure Python fallback will work but be slower.")

    score = max(0, min(100, 100 - 30 * len(blockers) - 8 * len(warnings)))
    status = "blocked" if blockers else ("ready" if score >= 85 else "partial")

    suggestions = [
        "python -m scripts.tools pretrained_status",
        "python -m scripts.tools benchmark_suite --help",
        "python -m scripts.tools auto_improve_loop --help",
        'python -m pytest -m "not cuda and not slow"',
    ]
    if dataset_manifest.strip():
        suggestions.append(f'python -m scripts.tools op_preflight --manifest "{dataset_path}"')

    return {
        "status": status,
        "score": score,
        "blockers": blockers,
        "warnings": warnings,
        "system": {"python": sys.version.split()[0], "platform": platform.platform(), "repo_root": str(repo_root)},
        "packages": pkgs,
        "gpu": gpu,
        "native": native,
        "native_presence": {"available": native_have, "total": native_total},
        "paths": {"key_files": key_files, "dataset_manifest_ok": dataset_ok, "dataset_manifest": dataset_path},
        "disk": disk,
        "suggested_next_commands": suggestions,
    }


def render_readiness_markdown(report: dict[str, Any]) -> str:
    np_ = report.get("native_presence", {}) or {}
    lines = [
        "# Startup Readiness Report",
        "",
        f"- Status: **{report.get('status', 'unknown')}**",
        f"- Score: **{int(report.get('score', 0) or 0)}/100**",
        f"- Native stack detected: **{int(np_.get('available', 0) or 0)}/{int(np_.get('total', 0) or 0)}**",
    ]
    for title, items in (("Blockers", report.get("blockers")), ("Warnings", report.get("warnings"))):
        lines += ["", f"## {title}"] + ([f"- {x}" for x in items] if items else ["- None"])
    lines += ["", "## Suggested Next Commands"]
    lines += [f"- `{cmd}`" for cmd in report.get("suggested_next_commands", []) or []]
    return "\n".join(lines) + "\n"


def _readiness_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="status readiness", description="Generate startup readiness report (no training run)."
    )
    ap.add_argument("--dataset-manifest", type=str, default="", help="Optional dataset manifest path to validate.")
    ap.add_argument("--work-dir", type=str, default="", help="Optional future run/work directory to check free space.")
    ap.add_argument("--out-json", type=str, default="startup_readiness_report.json")
    ap.add_argument("--out-md", type=str, default="", help="Optional markdown summary path.")
    args = ap.parse_args(argv)

    report = build_readiness_report(
        repo_root=ROOT, dataset_manifest=str(args.dataset_manifest), work_dir=str(args.work_dir)
    )
    out_path = _write_json(args.out_json, report)
    if str(args.out_md).strip():
        md_path = Path(str(args.out_md).strip())
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_readiness_markdown(report), encoding="utf-8")
    print(f"[startup_readiness] status={report['status']} score={report['score']} -> {out_path}")
    for title in ("blockers", "warnings"):
        if report[title]:
            print(f"[startup_readiness] {title}:")
            for b in report[title]:
                print(f"  - {b}")
    return 0


# ---------------------------------------------------------------- inventory (quality levers)

_INVENTORY_FILES = (
    ("quality_policy", "config/presets.py", "compete-mode soft defaults"),
    ("layout_attn_context", "models/attention.py", "Dense Diffusion thread state"),
    ("dense_diffusion", "frontier/layout.py", "bias_cross_attention"),
    ("hand_verifier", "utils/quality/critics.py", "pick-best hand heuristic"),
    ("editing_phase", "utils/generation/editing.py", "post-gen edit loop"),
    ("dpo_pipeline", "utils/agentic/quality_loop.py", "preference alignment"),
    ("flow_grpo", "utils/training/preference.py", "online RL hooks"),
    ("reference_token_projection", "models/conditioning.py", "IP-Adapter tokens"),
    ("frontier_consume", "utils/generation/guidance.py", "token emphasis + step_emphasis→noise_scales"),
)


def _collect_inventory() -> list[dict]:
    rows: list[dict] = []

    def add(name: str, status: str, path: str, note: str = "") -> None:
        rows.append({"name": name, "status": status, "path": path, "note": note})

    for name, rel, note in _INVENTORY_FILES:
        add(name, "present" if (ROOT / rel).is_file() else "missing", rel, note)

    try:
        from models.attention import clear_layout_attn, set_layout_attn

        clear_layout_attn()
        set_layout_attn(plan=None)
        add("layout_attn_api", "wired", "models/attention.py", "set/clear OK")
    except Exception as e:
        add("layout_attn_api", "broken", "models/attention.py", str(e)[:120])

    try:
        from config.presets import POLICY

        status = "wired" if POLICY.aggression == "max" else "partial"
        add("compete_mode", status, "config/presets.py", f"aggression={POLICY.aggression}")
    except Exception as e:
        add("compete_mode", "broken", "config/presets.py", str(e)[:120])

    add(
        "dense_diffusion_in_dit",
        "wired",
        "models/dit_text.py CrossAttention",
        "DiT_Text / DiT-P / Supreme read layout_attn_context during sample_loop",
    )
    add(
        "repa_default",
        "partial",
        "config/train_config.py",
        "REPA coded but repa_weight defaults to 0 — use train recipe repa_weight=0.5",
    )
    from scripts.tools.__main__ import commands

    ref = commands().get("train_reference_adapter")
    add(
        "reference_adapter_train",
        "present" if ref and ref[0].is_file() else "partial",
        "python -m scripts.tools train_reference_adapter",
        "train projector then pass --reference-adapter-pt",
    )
    return rows


def _inventory_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="status inventory", description="Inventory SDX quality levers")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args(argv)
    rows = _collect_inventory()
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


# ---------------------------------------------------------------- gen-searcher (Gen-Searcher output -> fact JSONL)


def _gen_searcher_main(argv: list[str]) -> int:
    from utils.modeling.model_paths import default_gen_searcher_8b_path, verify_gen_searcher_8b_local
    from utils.prompt.rag import load_facts_from_gen_searcher_json, merge_facts_into_prompt

    ap = argparse.ArgumentParser(
        prog="status gen-searcher",
        description="Convert Gen-Searcher-style JSON/JSONL outputs into SDX fact JSONL and optional merged prompt preview.",
    )
    ap.add_argument("--in-json", type=str, required=True, help="Gen-Searcher output path (.json or .jsonl).")
    ap.add_argument("--out-facts-jsonl", type=str, default="gen_searcher_facts.jsonl")
    ap.add_argument("--max-facts", type=int, default=24)
    ap.add_argument("--max-item-chars", type=int, default=280)
    ap.add_argument("--prompt", type=str, default="", help="Optional prompt to produce merged prompt preview.")
    ap.add_argument("--out-merged-prompt", type=str, default="", help="Optional path to write merged prompt text.")
    ap.add_argument("--max-merge-chars", type=int, default=2400)
    ap.add_argument(
        "--gen-searcher-model-path",
        type=str,
        default=default_gen_searcher_8b_path(),
        help="Local Gen-Searcher-8B folder or HF repo id (used for verification/status output).",
    )
    ap.add_argument(
        "--verify-local-model",
        action="store_true",
        help="Verify required local Gen-Searcher shard files before conversion.",
    )
    args = ap.parse_args(argv)

    if args.verify_local_model:
        report = verify_gen_searcher_8b_local(str(args.gen_searcher_model_path))
        print(f"[gen_searcher_bridge] model path: {args.gen_searcher_model_path}")
        print(
            f"[gen_searcher_bridge] local_ok={report['is_local_dir']} "
            f"all_required_present={report['all_required_present']}"
        )
        if report.get("missing"):
            print(f"[gen_searcher_bridge] missing: {report['missing']}")

    facts = load_facts_from_gen_searcher_json(
        args.in_json, max_entries=max(1, int(args.max_facts)), max_item_chars=max(64, int(args.max_item_chars))
    )
    out_path = Path(args.out_facts_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in facts), encoding="utf-8")
    print(f"[gen_searcher_bridge] wrote {len(facts)} facts -> {out_path}")

    if str(args.prompt).strip() and str(args.out_merged_prompt).strip():
        merged = merge_facts_into_prompt(str(args.prompt).strip(), facts, max_chars=max(256, int(args.max_merge_chars)))
        mp = Path(args.out_merged_prompt)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(merged, encoding="utf-8")
        print(f"[gen_searcher_bridge] wrote merged prompt -> {mp}")
    return 0


COMMANDS = {
    "pretrained": _pretrained_main,
    "readiness": _readiness_main,
    "inventory": _inventory_main,
    "gen-searcher": _gen_searcher_main,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("Commands: " + ", ".join(COMMANDS))
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    return int(COMMANDS[argv[0]](argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
