from pathlib import Path

from scripts.tools.ops import auto_improve_loop as ail


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
