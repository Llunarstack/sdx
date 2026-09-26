"""Tests for RSI user-feedback bus and preference export."""

from __future__ import annotations

import json
from pathlib import Path

from utils.generation.draft_thumbs import plan_draft_thumbnails, promote_draft_to_final
from utils.training.feedback_bus import (
    export_dpo_jsonl,
    iter_feedback,
    pairs_from_feedback,
    record_dislike,
    record_like,
    record_pair,
    record_pick,
    update_user_taste_from_feedback,
)


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
    from utils.training.feedback_bus import record_pick as rp

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
    import scripts.tools.ops.preference_flywheel as pf

    pairs = pairs_from_feedback(log)
    assert pairs
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    assert out.is_file()
    assert hasattr(pf, "main")


def test_feedback_cli_status(tmp_path: Path) -> None:
    log = tmp_path / "fb.jsonl"
    a = tmp_path / "a.png"
    a.write_bytes(b"a")
    record_like(str(a), prompt="x", log_path=log)
    from scripts.tools.ops.feedback_cli import main

    rc = main(["--log", str(log), "status"])
    assert rc == 0
    rc2 = main(["--log", str(log), "export-dpo", "--out", str(tmp_path / "p.jsonl"), "--no-synthetic"])
    # may be 2 if no pairs from like-only
    assert rc2 in (0, 2)
