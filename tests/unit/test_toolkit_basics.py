"""Smoke tests for toolkit package (QoL helpers)."""

from __future__ import annotations

import json
from pathlib import Path

def test_seed_everything_runs() -> None:
    from toolkit.training.seed_utils import seed_everything

    seed_everything(12345)
    seed_everything(42, deterministic_cudnn=False)


def test_worker_seed_fn_callable() -> None:
    from toolkit.training.seed_utils import worker_seed_fn

    fn = worker_seed_fn(99, rank=0)
    fn(0)
    fn(1)


def test_step_timer_ema() -> None:
    from toolkit.qol.timing import StepTimer

    t = StepTimer(ema_decay=0.5)
    t.tick_start()
    t.tick_end()
    t.tick_start()
    t.tick_end()
    assert t.steps == 2
    assert t.avg_sec > 0
    assert "s" in t.eta_str(1, 10)


def test_describe_optional_libs_keys() -> None:
    from toolkit.libs.optional_imports import describe_optional_libs

    d = describe_optional_libs()
    assert "xxhash" in d
    assert "installed" in d["xxhash"]


def test_manifest_digest_empty_jsonl(tmp_path: Path) -> None:
    from toolkit.quality.manifest_digest import digest_jsonl

    p = tmp_path / "m.jsonl"
    p.write_text("", encoding="utf-8")
    d = digest_jsonl(p)
    assert d["lines_read"] == 0


def test_manifest_digest_sample_rows(tmp_path: Path) -> None:
    from toolkit.quality.manifest_digest import digest_jsonl

    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"image_path": "a.png", "caption": "x"}) + "\n"
        + json.dumps({"path": "b.png", "text": "y"}) + "\n",
        encoding="utf-8",
    )
    d = digest_jsonl(p)
    assert d["lines_read"] == 2
    assert d["rows_with_image_field"] == 2
    assert d["rows_with_caption_like"] == 2


def test_collect_env_has_python() -> None:
    from toolkit.training.env_health import collect_env

    e = collect_env()
    assert "python" in e
    assert "platform" in e
