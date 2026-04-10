from __future__ import annotations

from pathlib import Path

import pytest
from utils.generation.eval_prompt_pack import load_eval_prompt_records, load_eval_prompts

_REPO_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "eval_prompts_baseline.json"


def test_load_baseline_example_pack() -> None:
    recs = load_eval_prompt_records(_REPO_EXAMPLE)
    prompts = load_eval_prompts(_REPO_EXAMPLE)
    assert len(recs) == len(prompts) >= 8
    assert all(r.prompt for r in recs)
    assert len({r.id for r in recs}) == len(recs)


def test_load_strings_only(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text('["  a ", "b"]', encoding="utf-8")
    assert load_eval_prompts(p) == ["a", "b"]


def test_reject_bad_root(tmp_path: Path) -> None:
    p = tmp_path / "p.json"
    p.write_text('{"prompt": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        load_eval_prompt_records(p)
