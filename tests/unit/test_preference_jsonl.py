"""Preference JSONL loader (no train integration)."""

import json
import tempfile
from pathlib import Path

from utils.training.preference_jsonl import iter_preference_jsonl, parse_preference_row


def test_parse_preference_row_minimal():
    p = parse_preference_row(
        {"win_image_path": "a.png", "lose_image_path": "b.png", "caption": "cat"}
    )
    assert p is not None
    assert p.win_path == "a.png" and p.lose_path == "b.png" and p.prompt == "cat"


def test_iter_preference_jsonl_skips_invalid():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "p.jsonl"
        f.write_text(
            json.dumps({"win_image_path": "w.jpg", "lose_image_path": "l.jpg"}) + "\n"
            "{not json}\n"
            + json.dumps({"only_win": "x"}) + "\n",
            encoding="utf-8",
        )
        rows = list(iter_preference_jsonl(f))
        assert len(rows) == 1
        assert rows[0].win_path.endswith("w.jpg")
