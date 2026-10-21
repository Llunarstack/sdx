"""

Also holds what used to be: test_eval_prompt_pack, test_eval_report.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from examples.run_baseline_eval import load_eval_prompts
from utils.superior.auto_loop import build_markdown_report

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
