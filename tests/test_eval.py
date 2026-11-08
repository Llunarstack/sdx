"""Also holds what used to be: test_eval_prompt_pack, test_eval_report, test_eval_harness, test_run_baseline_eval, test_benchmark_suite_tool, test_auto_improve_loop_tool."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from examples.run_baseline_eval import load_eval_prompts
from PIL import Image, ImageDraw
from scripts.tools import bench as bs
from scripts.tools import evals as scorers
from scripts.tools import superior as ail
from scripts.tools.evals import CATEGORIES, SUITE, evaluate, suite_as_manifest_rows
from utils.agentic.quality_loop import build_markdown_report

# ==================================================================================================
# test_eval_prompt_pack: test_eval_prompt_pack
# ==================================================================================================

_REPO_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "eval_prompts_baseline.json"


def test_load_baseline_example_pack() -> None:
    recs = load_eval_prompts(_REPO_EXAMPLE)
    assert len(recs) >= 8
    assert all(prompt for _, prompt in recs)
    assert len({eid for eid, _ in recs}) == len(recs)


def test_load_strings_only(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text('["  a ", "b"]', encoding="utf-8")
    assert load_eval_prompts(p) == [("p0", "a"), ("p1", "b")]


def test_reject_bad_root(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text('{"prompt": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        load_eval_prompts(p)


# ==================================================================================================
# test_eval_report: test_eval_report
# ==================================================================================================


def test_eval_report_includes_hf_section(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "results.json").write_text("[]", encoding="utf-8")
    (bench / "leaderboard.json").write_text("[]", encoding="utf-8")
    md = build_markdown_report(bench)
    assert "# SDX Benchmark Report" in md
    assert "## Leaderboard" in md


def test_eval_report_includes_text_encoder_stack(tmp_path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "results.json").write_text("[]", encoding="utf-8")
    (bench / "leaderboard.json").write_text("[]", encoding="utf-8")
    md = build_markdown_report(bench)
    assert "## Text encoder stack" in md or "## HF pretrained status" in md


# ==================================================================================================
# test_eval_harness: Tests for the competitor-weakness eval harness (scripts/tools/evals.py).
# ==================================================================================================


def test_suite_integrity():
    ids = [s.id for s in SUITE]
    assert len(ids) == len(set(ids)), "duplicate suite ids"
    for s in SUITE:
        assert s.category in CATEGORIES, f"{s.id}: unknown category {s.category}"
        if s.category == "count":
            assert s.expected_count is not None, f"{s.id}: count item needs expected_count"
        if s.category == "text_render":
            assert s.expected_text, f"{s.id}: text item needs expected_text"
    # Manifest export round-trips the objective fields.
    rows = suite_as_manifest_rows()
    assert len(rows) == len(SUITE)
    assert any("expected_count" in r for r in rows)
    assert any("expected_text" in r for r in rows)


def _discs(path: Path, n: int, bg="white", fg="red") -> None:
    im = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(im)
    xs = [int(256 * (i + 1) / (n + 1)) for i in range(n)]
    for cx in xs:
        d.ellipse([cx - 14, 118, cx + 14, 146], fill=fg)
    im.save(path)


def test_blob_counter_accuracy():
    with tempfile.TemporaryDirectory() as td:
        for n in (1, 2, 3, 5):
            p = Path(td) / f"n{n}.png"
            _discs(p, n)
            assert scorers.count_objects(str(p)) == n, f"expected {n} blobs"
        # Dark-background polarity is handled too.
        p = Path(td) / "dark.png"
        _discs(p, 4, bg="black", fg="yellow")
        assert scorers.count_objects(str(p)) == 4


def test_count_score_decay():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "three.png"
        _discs(p, 3)
        s_exact, d = scorers.count_score(str(p), 3)
        assert s_exact == 1.0 and d["detected"] == 3
        s_off, d2 = scorers.count_score(str(p), 5)  # detected 3, expected 5 -> err 2
        assert d2["abs_error"] == 2
        assert 0.0 < s_off < s_exact


def test_scene_physics_scorers_optional():
    # Must not raise when utils.quality.critics is absent.
    score, detail = scorers.occlusion_score("missing.png")
    assert score is None or isinstance(score, float)
    assert isinstance(detail, dict)
    score2, detail2 = scorers.reflection_score("missing.png")
    assert score2 is None or isinstance(score2, float)
    assert isinstance(detail2, dict)
    score3, detail3 = scorers.spatial_bind_score("missing.png", "a red cube to the left of a blue sphere")
    assert score3 is None or isinstance(score3, float)
    assert isinstance(detail3, dict)


def test_text_score_normalization():
    # LCS ratio helper is deterministic and offline.
    assert scorers._normalize_text("O.P-E N!") == "open"
    assert scorers._lcs_ratio("open", "the open sign") == 1.0
    assert scorers._lcs_ratio("open", "xxxx") == 0.0


def test_runner_aggregates():
    if not scorers.availability()["clip_adherence"]:
        import pytest

        pytest.skip("CLIP weights unavailable")
    with tempfile.TemporaryDirectory() as td:
        # Build images for one count item and one long-adherence item.
        count_item = next(s for s in SUITE if s.category == "count")
        long_item = next(s for s in SUITE if s.category == "long_adherence")
        cimg = Path(td) / "c.png"
        _discs(cimg, count_item.expected_count)
        limg = Path(td) / "l.png"
        Image.new("RGB", (256, 256), (30, 40, 90)).save(limg)
        report = evaluate(
            {count_item.id: str(cimg), long_item.id: str(limg)},
            clip_model="openai/clip-vit-base-patch32",
            device="cpu",
        )
        assert report["n_scored"] == 2
        assert report["overall_adherence"] is not None
        assert "count" in report["per_category"]
        # The count image matches expected exactly.
        assert report["per_category"]["count"]["count_score"] == 1.0


# ==================================================================================================
# test_run_baseline_eval
# ==================================================================================================


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def baseline_pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack.json"
    p.write_text('[{"id": "a", "prompt": "hello"}]', encoding="utf-8")
    return p


def test_run_baseline_eval_dry_run(capsys: pytest.CaptureFixture[str], baseline_pack: Path) -> None:
    ckpt = _REPO_ROOT / "README.md"
    argv = [
        "run_baseline_eval.py",
        "--ckpt",
        str(ckpt),
        "--pack",
        str(baseline_pack),
        "--out-dir",
        str(baseline_pack.parent / "out"),
    ]
    with mock.patch.object(sys, "argv", argv):
        from examples.run_baseline_eval import main

        assert main() == 0
    out = capsys.readouterr().out
    assert "WOULD RUN:" in out
    assert "sample.py" in out
    assert "hello" in out


# ==================================================================================================
# test_benchmark_suite_tool
# ==================================================================================================


def test_load_suite_default_non_empty():
    out = bs._load_suite(None, suite_pack="standard_v1")
    assert out
    assert any(c.expected_text for c in out)
    assert any(int(c.expected_count) > 0 for c in out)


def test_load_suite_biz_pack_non_empty():
    out = bs._load_suite(None, suite_pack="biz_visual_content_v1")
    assert out
    assert any(c.expected_text for c in out)


def test_load_suite_from_json(tmp_path: Path):
    p = tmp_path / "suite.json"
    p.write_text(
        json.dumps(
            [
                {
                    "name": "t1",
                    "prompt": 'store sign "OPEN"',
                    "expected_text": "OPEN",
                    "expected_count": 2,
                    "expected_count_target": "objects",
                    "expected_count_object": "coin",
                    "width": 640,
                    "height": 640,
                }
            ]
        ),
        encoding="utf-8",
    )
    out = bs._load_suite(p, suite_pack="standard_v1")
    assert len(out) == 1
    assert out[0].name == "t1"
    assert out[0].expected_text == "OPEN"
    assert out[0].expected_count == 2


def test_composite_score_uses_optional_metrics():
    case = bs.PromptCase(name="x", prompt="p", expected_text="ABC", expected_count=3)
    m = {
        "edge_sharpness": 200.0,
        "exposure_balance": 0.8,
        "saturation_balance": 0.9,
        "ocr_match": 1.0,
        "count_match": 0.5,
    }
    s = bs._composite_score(case, m)
    assert 0.0 <= s <= 1.0


def test_collect_checkpoints_from_dir(tmp_path: Path):
    d = tmp_path / "ckpts"
    d.mkdir(parents=True, exist_ok=True)
    a = d / "a.pt"
    b = d / "sub" / "b.safetensors"
    b.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    out = bs._collect_checkpoints([], str(d))
    assert str(a) in out
    assert str(b) in out


def test_write_preference_jsonl(tmp_path: Path):
    p1 = tmp_path / "m1.png"
    p2 = tmp_path / "m2.png"
    p1.write_bytes(b"x")
    p2.write_bytes(b"y")
    rows = [
        {"model": "m1", "case": "c1", "prompt": "p", "output": str(p1), "composite": 0.92},
        {"model": "m2", "case": "c1", "prompt": "p", "output": str(p2), "composite": 0.65},
    ]
    out_path = tmp_path / "prefs.jsonl"
    n = bs._write_preference_jsonl(out_path, rows, min_margin=0.1, max_pairs_per_case=2)
    assert n == 1
    txt = out_path.read_text(encoding="utf-8")
    assert "win_image_path" in txt


def test_parse_seeds_handles_list_and_fallback():
    assert bs._parse_seeds(42, "42, 7, bad, 99") == [42, 7, 99]
    assert bs._parse_seeds(42, "") == [42]


def test_aggregate_model_scores_applies_std_penalty():
    out = bs._aggregate_model_scores([0.9, 0.7, 0.8], robustness_penalty=0.5)
    assert abs(out["mean_composite"] - 0.8) < 1e-8
    assert out["std_composite"] > 0.0
    assert out["robust_score"] < out["mean_composite"]


def test_failure_tags_include_expected_failures():
    case = bs.PromptCase(name="c1", prompt='poster with text "OPEN"', expected_text="OPEN", expected_count=3)
    metrics = {
        "composite": 0.3,
        "ocr_match": 0.1,
        "count_match": 0.2,
        "exposure_balance": 0.4,
        "saturation_balance": 0.3,
        "edge_sharpness": 40.0,
    }
    tags = bs._failure_tags(case, metrics, threshold=0.6)
    assert "low_composite" in tags
    assert "text_rendering" in tags
    assert "counting" in tags
    assert "oversaturation" in tags


def test_write_hardcases_jsonl(tmp_path: Path):
    c1 = bs.PromptCase(name="c1", prompt='text "OPEN"', expected_text="OPEN")
    c2 = bs.PromptCase(name="c2", prompt="clean landscape")
    rows = [
        {
            "model": "m1",
            "case": "c1",
            "seed": 42,
            "prompt": c1.prompt,
            "output": "x.png",
            "composite": 0.4,
            "ocr_match": 0.2,
            "edge_sharpness": 50.0,
            "exposure_balance": 0.5,
            "saturation_balance": 0.5,
        },
        {
            "model": "m1",
            "case": "c2",
            "seed": 42,
            "prompt": c2.prompt,
            "output": "y.png",
            "composite": 0.85,
            "edge_sharpness": 300.0,
            "exposure_balance": 0.9,
            "saturation_balance": 0.9,
        },
    ]
    out_path = tmp_path / "hard.jsonl"
    n = bs._write_hardcases_jsonl(
        out_path,
        rows,
        {"c1": c1, "c2": c2},
        threshold=0.6,
        max_rows=10,
    )
    assert n == 1
    txt = out_path.read_text(encoding="utf-8")
    assert "failure_tags" in txt


# ==================================================================================================
# test_auto_improve_loop_tool
# ==================================================================================================


def test_best_model_tag():
    rows = [
        {"model": "base", "mean_composite": 0.72},
        {"model": "dpo_policy", "mean_composite": 0.81},
    ]
    assert ail._best_model_tag(rows) == "dpo_policy"


def test_resolve_ckpt_by_tag(tmp_path: Path):
    a = tmp_path / "best.pt"
    b = tmp_path / "dpo_policy.pt"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    got = ail._resolve_ckpt_by_tag([a, b], "dpo_policy")
    assert got == b


def test_read_leaderboard(tmp_path: Path):
    p = tmp_path / "leaderboard.json"
    p.write_text('[{"model":"m1","mean_composite":0.8}]', encoding="utf-8")
    out = ail._read_leaderboard(p)
    assert isinstance(out, list)
    assert out[0]["model"] == "m1"


def test_iter_dir_multi(tmp_path: Path):
    d = ail._iter_dir(tmp_path, 2, 5)
    assert d.name == "iter_002"


def test_iter_dir_single(tmp_path: Path):
    d = ail._iter_dir(tmp_path, 1, 1)
    assert d == tmp_path


def test_best_row():
    rows = [
        {"model": "base", "mean_composite": 0.72},
        {"model": "dpo_policy", "mean_composite": 0.81},
    ]
    br = ail._best_row(rows)
    assert br is not None
    assert br["model"] == "dpo_policy"
