from pathlib import Path

from scripts.tools.ops import startup_readiness as sr


def test_native_presence_score_counts_paths_and_bools():
    status = {
        "rust_sdx_jsonl_tools": "x.exe",
        "zig_sdx_linecrc": "",
        "go_sdx_manifest": "",
        "libsdx_latent": "/tmp/lib.dll",
        "libsdx_cuda_ml": "",
        "latent_lib_ctypes": True,  # not counted by key filter
    }
    have, total = sr._native_presence_score(status)
    assert have == 2
    assert total >= 5


def test_build_readiness_report_blocked_on_missing_required_packages(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sr, "_check_packages", lambda req, opt: {"required": {"torch": False}, "optional": {}})
    monkeypatch.setattr(sr, "_gather_native_status", lambda: {})
    monkeypatch.setattr(sr, "_gpu_status", lambda: {"torch_imported": False, "cuda_available": False})
    repo_root = tmp_path
    (repo_root / "sample.py").write_text("", encoding="utf-8")
    (repo_root / "train.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "tools" / "benchmark_suite.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "tools" / "ops" / "auto_improve_loop.py").write_text("", encoding="utf-8")
    out = sr.build_readiness_report(repo_root=repo_root)
    assert out["status"] == "blocked"
    assert out["blockers"]


def test_build_readiness_report_checks_dataset_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sr, "_check_packages", lambda req, opt: {"required": {"torch": True}, "optional": {}})
    monkeypatch.setattr(sr, "_gather_native_status", lambda: {})
    monkeypatch.setattr(sr, "_gpu_status", lambda: {"torch_imported": True, "cuda_available": True, "device_count": 1, "devices": []})
    repo_root = tmp_path
    (repo_root / "sample.py").write_text("", encoding="utf-8")
    (repo_root / "train.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "tools" / "benchmark_suite.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools" / "ops").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "tools" / "ops" / "auto_improve_loop.py").write_text("", encoding="utf-8")
    out = sr.build_readiness_report(repo_root=repo_root, dataset_manifest=str(tmp_path / "missing.jsonl"))
    assert out["paths"]["dataset_manifest_ok"] is False
    assert out["status"] == "blocked"


def test_render_readiness_markdown_contains_core_sections():
    report = {
        "status": "partial",
        "score": 77,
        "blockers": [],
        "warnings": ["example warning"],
        "native_presence": {"available": 3, "total": 10},
        "suggested_next_commands": ["python -m scripts.tools benchmark_suite --help"],
    }
    md = sr.render_readiness_markdown(report)
    assert "# Startup Readiness Report" in md
    assert "Status: **partial**" in md
    assert "## Suggested Next Commands" in md
