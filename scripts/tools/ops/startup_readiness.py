#!/usr/bin/env python3
"""
Pre-training readiness report for future startup rollout.

This does not train anything. It audits environment, native stack, key files,
and optional dataset/work paths, then writes a JSON report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _package_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _check_packages(required: List[str], optional: List[str]) -> Dict[str, Dict[str, bool]]:
    return {
        "required": {k: _package_available(k) for k in required},
        "optional": {k: _package_available(k) for k in optional},
    }


def _check_paths(paths: List[Path]) -> Dict[str, bool]:
    return {str(p): p.exists() for p in paths}


def _disk_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return 0.0
    return float(usage.free) / float(1024**3)


def _gpu_status() -> Dict[str, Any]:
    out: Dict[str, Any] = {"torch_imported": False, "cuda_available": False, "device_count": 0, "devices": []}
    try:
        import torch  # type: ignore
    except Exception:
        return out
    out["torch_imported"] = True
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    out["cuda_available"] = cuda_ok
    if not cuda_ok:
        return out
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    out["device_count"] = n
    devices = []
    for i in range(max(0, n)):
        row: Dict[str, Any] = {"index": i}
        try:
            row["name"] = str(torch.cuda.get_device_name(i))
        except Exception:
            row["name"] = ""
        try:
            props = torch.cuda.get_device_properties(i)
            row["total_vram_gb"] = float(props.total_memory) / float(1024**3)
        except Exception:
            row["total_vram_gb"] = 0.0
        devices.append(row)
    out["devices"] = devices
    return out


def _gather_native_status() -> Dict[str, Any]:
    try:
        from utils.native.native_tools import native_stack_status

        return dict(native_stack_status())
    except Exception as exc:
        return {"error": str(exc)}


def _native_presence_score(native_status: Dict[str, Any]) -> Tuple[int, int]:
    keys = [k for k in native_status.keys() if k.startswith("rust_") or k.startswith("zig_") or k.startswith("go_")]
    keys += [k for k in native_status.keys() if k.startswith("libsdx_")]
    keys = sorted(set(keys))
    if not keys:
        return 0, 0
    have = 0
    for k in keys:
        v = native_status.get(k)
        if isinstance(v, bool):
            have += int(v)
        elif isinstance(v, str):
            have += int(bool(v.strip()))
    return have, len(keys)


def build_readiness_report(
    *,
    repo_root: Path,
    dataset_manifest: str = "",
    work_dir: str = "",
) -> Dict[str, Any]:
    required_pkgs = ["numpy", "PIL", "torch"]
    optional_pkgs = ["diffusers", "transformers", "cv2", "pytesseract"]
    pkgs = _check_packages(required_pkgs, optional_pkgs)
    gpu = _gpu_status()
    native = _gather_native_status()

    key_files = _check_paths(
        [
            repo_root / "sample.py",
            repo_root / "train.py",
            repo_root / "scripts" / "tools" / "benchmark_suite.py",
            repo_root / "scripts" / "tools" / "ops" / "auto_improve_loop.py",
        ]
    )

    disk_paths = [repo_root]
    if work_dir.strip():
        disk_paths.append(Path(work_dir).resolve())
    disk = {str(p): {"free_gb": _disk_free_gb(p)} for p in disk_paths}

    dataset_ok = True
    dataset_path = ""
    if dataset_manifest.strip():
        dataset_path = str(Path(dataset_manifest).resolve())
        dataset_ok = Path(dataset_path).is_file()

    blockers: List[str] = []
    warnings: List[str] = []

    if not all(pkgs["required"].values()):
        missing = [k for k, ok in pkgs["required"].items() if not ok]
        blockers.append(f"Missing required packages: {', '.join(missing)}")
    if not all(key_files.values()):
        missing_files = [k for k, ok in key_files.items() if not ok]
        blockers.append(f"Missing key files: {', '.join(missing_files)}")
    if not dataset_ok:
        blockers.append(f"Dataset manifest not found: {dataset_path}")

    if not bool(gpu.get("cuda_available", False)):
        warnings.append("CUDA not detected; future training/inference will be slower on CPU.")

    low_disk = [k for k, row in disk.items() if float(row.get("free_gb", 0.0)) < 20.0]
    if low_disk:
        warnings.append(f"Low disk headroom (<20GB free): {', '.join(low_disk)}")

    native_have, native_total = _native_presence_score(native)
    if native_total > 0 and native_have == 0:
        warnings.append("No native accelerators detected; pure Python fallback will work but be slower.")

    score = 100
    score -= 30 * len(blockers)
    score -= 8 * len(warnings)
    score = max(0, min(100, score))

    if blockers:
        status = "blocked"
    elif score >= 85:
        status = "ready"
    else:
        status = "partial"

    suggestions = [
        "python -m scripts.tools pretrained_status",
        "python -m scripts.tools benchmark_suite --help",
        "python -m scripts.tools auto_improve_loop --help",
        "python -m pytest -m \"not cuda and not slow\"",
    ]
    if dataset_manifest.strip():
        suggestions.append(
            f"python -m scripts.tools op_preflight --manifest \"{dataset_path}\""
        )

    return {
        "status": status,
        "score": score,
        "blockers": blockers,
        "warnings": warnings,
        "system": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "repo_root": str(repo_root),
        },
        "packages": pkgs,
        "gpu": gpu,
        "native": native,
        "native_presence": {"available": native_have, "total": native_total},
        "paths": {"key_files": key_files, "dataset_manifest_ok": dataset_ok, "dataset_manifest": dataset_path},
        "disk": disk,
        "suggested_next_commands": suggestions,
    }


def render_readiness_markdown(report: Dict[str, Any]) -> str:
    status = str(report.get("status", "unknown"))
    score = int(report.get("score", 0) or 0)
    blockers = list(report.get("blockers", []) or [])
    warnings = list(report.get("warnings", []) or [])
    native_presence = report.get("native_presence", {}) or {}
    np_have = int(native_presence.get("available", 0) or 0)
    np_total = int(native_presence.get("total", 0) or 0)
    lines = [
        "# Startup Readiness Report",
        "",
        f"- Status: **{status}**",
        f"- Score: **{score}/100**",
        f"- Native stack detected: **{np_have}/{np_total}**",
        "",
        "## Blockers",
    ]
    if blockers:
        lines.extend([f"- {b}" for b in blockers])
    else:
        lines.append("- None")
    lines += ["", "## Warnings"]
    if warnings:
        lines.extend([f"- {w}" for w in warnings])
    else:
        lines.append("- None")
    lines += ["", "## Suggested Next Commands"]
    for cmd in list(report.get("suggested_next_commands", []) or []):
        lines.append(f"- `{cmd}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate startup readiness report (no training run).")
    ap.add_argument("--dataset-manifest", type=str, default="", help="Optional dataset manifest path to validate.")
    ap.add_argument("--work-dir", type=str, default="", help="Optional future run/work directory to check free space.")
    ap.add_argument("--out-json", type=str, default="startup_readiness_report.json")
    ap.add_argument("--out-md", type=str, default="", help="Optional markdown summary path.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    report = build_readiness_report(
        repo_root=repo_root,
        dataset_manifest=str(args.dataset_manifest),
        work_dir=str(args.work_dir),
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if str(args.out_md).strip():
        md_path = Path(str(args.out_md).strip())
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_readiness_markdown(report), encoding="utf-8")
    print(f"[startup_readiness] status={report['status']} score={report['score']} -> {out_path}")
    if report["blockers"]:
        print("[startup_readiness] blockers:")
        for b in report["blockers"]:
            print(f"  - {b}")
    if report["warnings"]:
        print("[startup_readiness] warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
