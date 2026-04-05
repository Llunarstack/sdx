from pathlib import Path

from scripts.tools.training import mine_preference_pairs as mpp


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
