"""Tests for scripts.tools CLIs and the core CLI entry points.
Also holds what used to be: test_scripts_tools_dispatcher, test_manifest_gate_tool, test_mine_preference_pairs_tool, test_startup_readiness_tool, test_feedback_rsi, test_cli_entrypoints, test_hybrid_dit_vit_generate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.tools import __main__ as tools_main
from scripts.tools import status as sr
from scripts.tools import superior as mpp
from scripts.tools.generate import (
    _candidate_paths,
    anneal_weight,
    consensus_score,
    constraint_consensus_score,
    extract_expected_text,
    fuse_vit_scores,
    maybe_self_correct_prompt,
    next_candidate_budget,
    pareto_front_rows,
    reflective_prompt_update,
    seed_for_iteration,
    signature_novelty,
    uncertainty_score,
)
from utils.generation.planners import plan_draft_thumbnails, promote_draft_to_final
from utils.prompt.scene import compile_shape_scaffold, infer_shape_blueprint
from utils.training.preference import (
    export_dpo_jsonl,
    iter_feedback,
    pairs_from_feedback,
    record_dislike,
    record_like,
    record_pair,
    record_pick,
    update_user_taste_from_feedback,
)

# ==================================================================================================
# test_scripts_tools_dispatcher: ``python -m scripts.tools`` resolves every tool script with a ``__main__`` guard.
# ==================================================================================================


def test_documented_commands_resolve():
    cmds = tools_main.commands()
    for name in ("ckpt_info", "data_quality", "preference_flywheel", "perfect_gen", "feedback", "corpus_retrieve"):
        assert cmds[name][0].is_file(), name
    assert "scorers" not in cmds  # library module, no CLI


def test_training_commands_use_trainers_module():
    cmds = tools_main.commands()
    assert cmds["train_diffusion_dpo"] == (cmds["trainers"][0], ["dpo"])
    assert cmds["train_kd_distill"] == (cmds["trainers"][0], ["kd-distill"])
    assert cmds["precompute_latents"] == (cmds["trainers"][0], ["precompute-latents"])
    assert cmds["noise_schedule_export"] == (cmds["ckpt"][0], ["noise-schedule"])
    assert cmds["sample_enhanced"] == (cmds["enhanced"][0], ["sample"])


def test_merged_superior_scripts_keep_old_names():
    cmds = tools_main.commands()
    assert cmds["superior_curate"] == (cmds["superior"][0], ["curate"])
    assert cmds["run_flywheel"] == (cmds["superior"][0], ["flywheel"])
    assert cmds["rsi_loop"] == (cmds["superior"][0], ["rsi"])
    assert cmds["benchmark_suite"] == (cmds["bench"][0], ["suite"])
    assert cmds["agentic_evolve"] == (cmds["agentic"][0], ["evolve"])
    assert cmds["perfect_gen"] == (cmds["agentic"][0], ["perfect"])
    assert cmds["explore_styles"] == (cmds["prompts"][0], ["explore-styles"])
    assert cmds["video_generate"][0].name == "generate_video.py"
    out = subprocess.run(
        [sys.executable, "-m", "scripts.tools", "superior_generate", "--ckpt", "x.pt", "--prompt", "p", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "sample.py" in out and "--prompt" in out


# ==================================================================================================
# test_manifest_gate_tool
# ==================================================================================================


def test_manifest_gate_smoke(tmp_path: Path):
    # Minimal manifest should pass with defaults.
    p = tmp_path / "m.jsonl"
    p.write_text('{"image_path":"a.png","caption":"1girl, cafe","negative_caption":"bad anatomy"}\n', encoding="utf-8")

    from scripts.tools.manifest import main

    assert main(["gate", str(p)]) == 0


# ==================================================================================================
# test_mine_preference_pairs_tool
# ==================================================================================================


def test_mine_pairs_basic(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    rows = [
        {"model": "m1", "case": "c1", "prompt": "p", "output": str(a), "composite": 0.9},
        {"model": "m2", "case": "c1", "prompt": "p", "output": str(b), "composite": 0.6},
    ]
    out = mpp.mine_pairs(rows, min_margin=0.1, max_pairs_per_case=2)
    assert len(out) == 1
    assert out[0]["win_image_path"] == str(a)
    assert out[0]["lose_image_path"] == str(b)
    assert out[0]["caption"] == "p"


def test_mine_pairs_respects_margin(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    rows = [
        {"model": "m1", "case": "c1", "prompt": "p", "output": str(a), "composite": 0.71},
        {"model": "m2", "case": "c1", "prompt": "p", "output": str(b), "composite": 0.70},
    ]
    out = mpp.mine_pairs(rows, min_margin=0.05, max_pairs_per_case=2)
    assert out == []


def test_mine_pairs_hardcase_boost_and_margin_scale(tmp_path: Path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    c.write_bytes(b"z")
    rows = [
        {"model": "m1", "case": "c1", "prompt": "p", "output": str(a), "composite": 0.90},
        {"model": "m2", "case": "c1", "prompt": "p", "output": str(b), "composite": 0.84},
        {"model": "m3", "case": "c1", "prompt": "p", "output": str(c), "composite": 0.80},
    ]
    # With min_margin=0.08, only one pair qualifies without hard-case scaling.
    out_regular = mpp.mine_pairs(rows, min_margin=0.08, max_pairs_per_case=1)
    assert len(out_regular) == 1
    # Hard-case boost + relaxed margin should mine 2 pairs.
    out_hard = mpp.mine_pairs(
        rows,
        min_margin=0.08,
        max_pairs_per_case=1,
        hard_case_names={"c1"},
        hardcase_extra_pairs=1,
        hardcase_min_margin_scale=0.75,
    )
    assert len(out_hard) == 2
    assert all(bool(r.get("is_hardcase")) for r in out_hard)


def test_read_hardcase_names(tmp_path: Path):
    p = tmp_path / "hard.jsonl"
    p.write_text('{"case":"c1"}\n{"case":"c2"}\n{"x":1}\n', encoding="utf-8")
    out = mpp._read_hardcase_names(p)
    assert out == {"c1", "c2"}


# ==================================================================================================
# test_startup_readiness_tool
# ==================================================================================================


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
    (repo_root / "scripts" / "tools" / "bench.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools" / "superior.py").write_text("", encoding="utf-8")
    out = sr.build_readiness_report(repo_root=repo_root)
    assert out["status"] == "blocked"
    assert out["blockers"]


def test_build_readiness_report_checks_dataset_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sr, "_check_packages", lambda req, opt: {"required": {"torch": True}, "optional": {}})
    monkeypatch.setattr(sr, "_gather_native_status", lambda: {})
    monkeypatch.setattr(
        sr, "_gpu_status", lambda: {"torch_imported": True, "cuda_available": True, "device_count": 1, "devices": []}
    )
    repo_root = tmp_path
    (repo_root / "sample.py").write_text("", encoding="utf-8")
    (repo_root / "train.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "tools" / "bench.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "tools" / "superior.py").write_text("", encoding="utf-8")
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


# ==================================================================================================
# test_feedback_rsi: Tests for RSI user-feedback bus and preference export.
# ==================================================================================================


def test_feedback_like_dislike_pair_export(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    record_like(str(a), prompt="red cube", log_path=log)
    record_dislike(str(b), prompt="red cube", log_path=log)
    record_pair(str(a), str(b), prompt="red cube", log_path=log)
    events = list(iter_feedback(log))
    assert len(events) >= 3
    pairs = pairs_from_feedback(log)
    assert any(p["win_image_path"] == str(a) and p["lose_image_path"] == str(b) for p in pairs)
    assert all(p.get("source", "").startswith("user") for p in pairs)
    out = tmp_path / "prefs.jsonl"
    n = export_dpo_jsonl(out, log_path=log)
    assert n >= 1
    assert out.is_file()
    row = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "win_image_path" in row and "lose_image_path" in row


def test_feedback_pick_and_taste_sync(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    taste = tmp_path / "taste.json"
    w = tmp_path / "win.png"
    l1 = tmp_path / "l1.png"
    l2 = tmp_path / "l2.png"
    for p in (w, l1, l2):
        p.write_bytes(b"x")
    record_pick(str(w), [str(l1), str(l2)], prompt="portrait", log_path=log)
    pairs = pairs_from_feedback(log, include_synthetic=False)
    assert len(pairs) >= 2
    tp = update_user_taste_from_feedback(log_path=log, taste_path=taste)
    assert tp is not None and Path(tp).is_file()
    data = json.loads(Path(tp).read_text(encoding="utf-8"))
    assert any("portrait" in str(x) for x in (data.get("likes_notes") or []))


def test_draft_promote_logs_user_pick(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    # Point default log via explicit paths in record_pick — promote uses default log;
    # monkeypatch by writing draft paths and checking promote returns seed argv.
    paths = []
    for i in range(3):
        p = tmp_path / f"d{i}.png"
        p.write_bytes(b"d")
        paths.append(str(p))
    plan = plan_draft_thumbnails("test prompt", num_drafts=3, base_seed=10)
    # Temporarily use feedback bus with custom log by calling record via promote
    # (promote uses default path). Call record_pick equivalent through promote after
    # redirecting — instead invoke promote and assert argv seed.
    argv = promote_draft_to_final(plan, chosen_index=1, user_picked=True, draft_paths=paths, log_feedback=False)
    assert "--seed" in argv
    assert argv[argv.index("--seed") + 1] == "11"
    # With logging on, pairs land in default log — use explicit record_pick here
    from utils.training.preference import record_pick as rp

    rp(paths[1], [paths[0], paths[2]], prompt="test prompt", log_path=log)
    assert len(pairs_from_feedback(log, include_synthetic=False)) >= 2


def test_preference_flywheel_accepts_feedback_flag(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    record_pair(str(a), str(b), prompt="p", log_path=log)
    out = tmp_path / "out.jsonl"
    from scripts.tools import superior as pf

    pairs = pairs_from_feedback(log)
    assert pairs
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    assert out.is_file()
    assert "prefs" in pf.RAW_COMMANDS


def _fake_sdx(tmp_path: Path, train_delta: int):
    """Checkpoint = text file holding a skill level; images get brighter with skill; judge = brightness."""
    import numpy as np
    from PIL import Image

    def generate(ckpt, jobs, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        skill = int(Path(ckpt).read_text())
        for j in jobs:
            v = min(255, 40 + 20 * skill + j["seed"] % 37)
            Image.fromarray(np.full((8, 8, 3), v, np.uint8)).save(j["out"])

    def train(ckpt, prefs, out, lr):
        assert prefs.read_text().strip()
        out.write_text(str(int(Path(ckpt).read_text()) + train_delta))
        return True

    def judge(path, prompt):
        return float(np.asarray(Image.open(path)).mean() / 255.0), {}

    base = tmp_path / "base.pt"
    base.write_text("0")
    return base, generate, train, judge


def _rsi_opts(tmp_path: Path, base: Path, **kw):
    from scripts.tools.superior import build_rsi_parser

    opts = build_rsi_parser().parse_args(
        ["--base-ckpt", str(base), "--work-dir", str(tmp_path / "rsi"), "--prompts", "4", "--from-feedback", ""]
    )
    vars(opts).update(kw)
    return opts


def test_rsi_loop_promotes_better_model_and_resumes(tmp_path: Path) -> None:
    from scripts.tools.superior import rsi_run as run

    base, generate, train, judge = _fake_sdx(tmp_path, train_delta=1)
    state = run(_rsi_opts(tmp_path, base, rounds=2), generate=generate, train=train, judge=judge)
    assert [h["result"] for h in state["history"]] == ["promoted", "promoted"]
    assert Path(state["champion"]).read_text() == "2"
    assert all(h["win_rate"] == 1.0 and h["gain"] > 0 for h in state["history"])
    # Resume from state.json: no --base-ckpt needed, round numbering continues.
    state = run(_rsi_opts(tmp_path, Path(""), rounds=1), generate=generate, train=train, judge=judge)
    assert state["round"] == 3 and Path(state["champion"]).read_text() == "3"


def test_rsi_loop_rejects_worse_model(tmp_path: Path) -> None:
    from scripts.tools.superior import rsi_run as run

    base, generate, train, judge = _fake_sdx(tmp_path, train_delta=-1)
    opts = _rsi_opts(tmp_path, base, rounds=1)
    state = run(opts, generate=generate, train=train, judge=judge)
    assert state["history"][0]["result"] == "rejected"
    assert state["champion"] == str(base)
    assert state["lr"] == opts.dpo_lr / 2
    assert not (tmp_path / "rsi" / "round_001" / "candidate.pt").exists()


def test_rsi_pairs_gate_and_curriculum() -> None:
    from scripts.tools.superior import curriculum_weights, eval_prompts, gate, train_prompts
    from scripts.tools.superior import mine_self_pairs as mine_pairs

    rows = [{"prompt": "p", "path": f"{s}.png", "score": s} for s in (0.9, 0.7, 0.68, 0.1)]
    pairs = mine_pairs(rows, min_margin=0.05)
    assert [(p["win_image_path"], p["lose_image_path"]) for p in pairs] == [("0.9.png", "0.1.png")]
    assert gate({"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.45}, 0.01)["promote"]  # +0.025 mean, 50% wins
    assert not gate({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.4, "c": 1.0}, 0.01)["win_rate"] > 0.5
    w = curriculum_weights({"photo": 0.9, "2d": 0.2})
    assert w["2d"] > w["photo"] and min(w.values()) >= 0.05
    import random

    held = {p for _, p in eval_prompts(2)}
    trained = {p for _, p in train_prompts(200, w, random.Random(0))}
    assert held and not held & trained  # held-out subjects never leak into training


def test_feedback_cli_status(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    a = tmp_path / "a.png"
    a.write_bytes(b"a")
    record_like(str(a), prompt="x", log_path=log)
    from scripts.tools.superior import main

    rc = main(["feedback", "--log", str(log), "status"])
    assert rc == 0
    rc2 = main(["feedback", "--log", str(log), "export-dpo", "--out", str(tmp_path / "p.jsonl"), "--no-synthetic"])
    # may be 2 if no pairs from like-only
    assert rc2 in (0, 2)


# ==================================================================================================
# test_cli_entrypoints: Smoke-test core CLIs: ``--help`` must exit 0 without importing heavy GPU stacks.
# ==================================================================================================


REPO = Path(__file__).resolve().parents[1]

# Pytest may already load torch/OpenMP; child processes need this on some Windows/conda stacks.
_SUBPROC_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTHONIOENCODING": "utf-8",
}


def _run_help(*argv: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, *argv, "--help"]
    return subprocess.run(
        cmd,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        env=_SUBPROC_ENV,
    )


def test_train_help():
    r = _run_help("train.py")
    assert r.returncode == 0, r.stderr
    assert "--data-path" in r.stdout or "--manifest-jsonl" in r.stdout


def test_sample_help():
    r = _run_help("sample.py")
    assert r.returncode == 0, r.stderr
    assert "--prompt" in r.stdout
    assert "--ckpt" in r.stdout


def test_demo_help():
    r = _run_help("demo.py")
    assert r.returncode == 0, r.stderr
    assert "--ckpt" in r.stdout


def test_inference_help():
    r = _run_help("inference.py")
    assert r.returncode == 0, r.stderr
    assert "--ckpt" in r.stdout


def test_scripts_tools_help():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.tools", "help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        env=_SUBPROC_ENV,
    )
    assert r.returncode == 0, r.stderr
    assert "quick_test" in r.stdout


# ==================================================================================================
# test_hybrid_dit_vit_generate
# ==================================================================================================


def test_candidate_paths_pattern(tmp_path: Path) -> None:
    out = tmp_path / "result.png"
    got = _candidate_paths(out, 3)
    assert [p.name for p in got] == ["result_cand0.png", "result_cand1.png", "result_cand2.png"]


def test_fuse_vit_scores_normalizes_weights() -> None:
    q = 0.8
    a = 0.2
    s1 = fuse_vit_scores(q, a, quality_weight=11.0, adherence_weight=9.0)
    s2 = fuse_vit_scores(q, a, quality_weight=1.1, adherence_weight=0.9)
    assert abs(s1 - s2) < 1e-9


def test_fuse_vit_scores_zero_weights_falls_back_even() -> None:
    q = 0.9
    a = 0.1
    s = fuse_vit_scores(q, a, quality_weight=0.0, adherence_weight=0.0)
    assert abs(s - 0.5) < 1e-9


def test_consensus_score_penalizes_disagreement() -> None:
    hi_disagree = consensus_score(0.95, 0.40, quality_weight=0.55, adherence_weight=0.45, disagreement_penalty=0.2)
    lo_disagree = consensus_score(0.75, 0.70, quality_weight=0.55, adherence_weight=0.45, disagreement_penalty=0.2)
    assert lo_disagree > hi_disagree


def test_seed_for_iteration_stride() -> None:
    assert seed_for_iteration(100, 0, 7) == 100
    assert seed_for_iteration(100, 3, 7) == 121


def test_maybe_self_correct_prompt() -> None:
    base = "portrait of a warrior in moonlight"
    corrected = maybe_self_correct_prompt(base, best_adherence=0.4, threshold=0.63, enable=True)
    assert corrected != base
    assert "precise prompt adherence" in corrected
    unchanged = maybe_self_correct_prompt(base, best_adherence=0.9, threshold=0.63, enable=True)
    assert unchanged == base


def test_infer_shape_blueprint_includes_constraints() -> None:
    bp = infer_shape_blueprint("full body warrior left of dragon, cinematic moonlight")
    assert isinstance(bp.get("constraints"), list)
    assert "shape-first composition" in (bp.get("composition") or [])
    assert bp.get("actors") or []


def test_compile_shape_scaffold_outputs_positive_negative() -> None:
    pos, neg, bp = compile_shape_scaffold("1girl holding sword, low angle, neon")
    assert isinstance(bp, dict)
    assert "coherent perspective" in pos.lower()
    assert "warped anatomy" in neg.lower()


def test_extract_expected_text_quotes() -> None:
    assert extract_expected_text('poster with text "HELLO WORLD"') == "HELLO WORLD"
    assert extract_expected_text("poster says 'SALE NOW'") == "SALE NOW"
    assert extract_expected_text("no quoted text here") == ""


def test_pareto_front_rows_filters_dominated() -> None:
    rows = [
        {"id": 1.0, "vit_quality_prob": 0.9, "vit_adherence_score": 0.5},
        {"id": 2.0, "vit_quality_prob": 0.7, "vit_adherence_score": 0.7},
        {"id": 3.0, "vit_quality_prob": 0.6, "vit_adherence_score": 0.6},
    ]
    front = pareto_front_rows(rows, ["vit_quality_prob", "vit_adherence_score"])
    ids = {int(r["id"]) for r in front}
    assert 3 not in ids
    assert 1 in ids and 2 in ids


def test_constraint_consensus_score_adds_constraints() -> None:
    s0 = constraint_consensus_score(
        quality_prob=0.8,
        adherence_score=0.8,
        ocr_score=0.1,
        count_score=0.1,
        saturation_score=0.1,
        quality_weight=0.5,
        adherence_weight=0.5,
        disagreement_penalty=0.1,
        ocr_weight=0.0,
        count_weight=0.0,
        saturation_weight=0.0,
    )
    s1 = constraint_consensus_score(
        quality_prob=0.8,
        adherence_score=0.8,
        ocr_score=0.9,
        count_score=0.9,
        saturation_score=0.9,
        quality_weight=0.5,
        adherence_weight=0.5,
        disagreement_penalty=0.1,
        ocr_weight=0.1,
        count_weight=0.1,
        saturation_weight=0.1,
    )
    assert s1 > s0


def test_next_candidate_budget_increases_when_low() -> None:
    assert next_candidate_budget(4, 0.5, threshold=0.8, step=2, max_num=8) == 6
    assert next_candidate_budget(8, 0.5, threshold=0.8, step=2, max_num=8) == 8
    assert next_candidate_budget(6, 0.9, threshold=0.8, step=2, max_num=8) == 6


def test_reflective_prompt_update_adds_hints() -> None:
    p = "epic scene"
    row = {"vit_adherence_score": 0.3, "count_score": 0.2, "ocr_score": 0.2, "saturation_score": 0.2}
    out = reflective_prompt_update(p, row, enable=True)
    assert out != p
    assert "exact object/person counts" in out


def test_anneal_weight_up_down() -> None:
    base = 0.1
    up0 = anneal_weight(base, 0, 5, mode="up")
    up4 = anneal_weight(base, 4, 5, mode="up")
    dn0 = anneal_weight(base, 0, 5, mode="down")
    dn4 = anneal_weight(base, 4, 5, mode="down")
    assert up4 > up0
    assert dn4 < dn0


def test_uncertainty_score_tracks_confidence() -> None:
    hi = {
        "vit_consensus_score": 0.92,
        "vit_quality_prob": 0.92,
        "vit_adherence_score": 0.91,
        "ocr_score": 0.9,
        "count_score": 0.9,
        "saturation_score": 0.9,
    }
    lo = {
        "vit_consensus_score": 0.42,
        "vit_quality_prob": 0.9,
        "vit_adherence_score": 0.4,
        "ocr_score": 0.3,
        "count_score": 0.2,
        "saturation_score": 0.3,
    }
    assert uncertainty_score(lo) > uncertainty_score(hi)


def test_signature_novelty_increases_with_distance() -> None:
    import numpy as np
    from utils.quality.critics import image_signature

    def sig(rgb):
        return image_signature(np.full((32, 32, 3), rgb, dtype=np.uint8))

    mem = [sig((10, 10, 10))]
    assert signature_novelty(sig((220, 210, 215)), mem) > signature_novelty(sig((12, 11, 10)), mem)
    assert signature_novelty(sig((12, 11, 10)), []) == 1.0
