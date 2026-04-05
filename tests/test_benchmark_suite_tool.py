import json
from pathlib import Path

from scripts.tools import benchmark_suite as bs


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
