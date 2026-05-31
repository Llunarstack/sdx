"""Data quality pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

from utils.data_quality import FilterConfig, filter_jsonl_file


def test_filter_jsonl_caption_length(tmp_path: Path) -> None:
    inp = tmp_path / "m.jsonl"
    rows = [
        {"image_path": "a.png", "caption": "short"},
        {"image_path": "b.png", "caption": "this caption is long enough to pass"},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    kept, stats = filter_jsonl_file(
        inp,
        config=FilterConfig(min_caption_len=10, dedup=""),
        output_path=out,
    )
    assert stats.kept == 1
    assert kept[0]["caption"].startswith("this caption")
