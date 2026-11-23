"""Tests for utils.runtime and utils.native fast paths.
Also holds what used to be: test_jsonutil, test_runtime_profiling, test_native_fast_paths, test_plain_dict_snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from utils.runtime import ProfileConfig, consume_profile_args, json_dumps, json_loads, run_with_cprofile, to_plain_dict

from sdx_native.native_tools import (
    SHARED_LIBRARIES,
    fnv1a64_file,
    manifest_fingerprint_line,
    merge_jsonl_files,
    native_stack_status,
)

# ==================================================================================================
# test_jsonutil
# ==================================================================================================


def test_loads_roundtrip_stdlib():
    assert json_loads('{"a": 1}') == {"a": 1}
    assert json_loads(b'{"b": 2}') == {"b": 2}


def test_dumps_utf8_fast_path_matches_json():
    obj = {"x": 1, "ü": "beta"}
    a = json_dumps(obj, ensure_ascii=False, sort_keys=True)
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    assert json.loads(a) == json.loads(b)


def test_dumps_ensure_ascii_uses_stdlib_escape():
    obj = {"ü": "x"}
    s = json_dumps(obj, ensure_ascii=True, sort_keys=True)
    assert "\\u" in s or "ü" not in s


def test_loads_accepts_bytes():
    assert json_loads(b'{"k": null}') == {"k": None}


def test_dumps_with_default_forces_stdlib():
    dt = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    s = json_dumps({"t": dt}, default=lambda o: o.isoformat(), ensure_ascii=False)
    assert json.loads(s)["t"].startswith("2024-06-01")


def test_dumps_indent_true_maps_to_two_spaces():
    obj = {"a": [1, 2]}
    s = json_dumps(obj, indent=True, ensure_ascii=False, sort_keys=True)
    parsed = json.loads(s)
    assert parsed == obj
    assert "\n" in s


# ==================================================================================================
# test_runtime_profiling
# ==================================================================================================


def test_consume_profile_args_strips_flags():
    argv = ["prog", "--profile-out", "x.prof", "--profile-sort", "tottime", "--profile-top", "10", "--ckpt", "m.pt"]
    rest, cfg = consume_profile_args(argv)
    assert rest == ["prog", "--ckpt", "m.pt"]
    assert cfg is not None
    assert cfg.out_path == "x.prof"
    assert cfg.sort_key == "tottime"
    assert cfg.top_n == 10


def test_consume_profile_args_no_profile():
    argv = ["prog", "--ckpt", "m.pt"]
    rest, cfg = consume_profile_args(argv)
    assert rest == argv
    assert cfg is None


def test_run_with_cprofile_writes_files(tmp_path):
    out = tmp_path / "run.prof"
    cfg = ProfileConfig(out_path=str(out), sort_key="cumulative", top_n=5)

    def _work():
        x = 0
        for i in range(1000):
            x += i

    run_with_cprofile(_work, cfg)
    assert out.is_file()
    summary = tmp_path / "run.prof.txt"
    assert summary.is_file()
    assert summary.stat().st_size > 0


# ==================================================================================================
# test_native_fast_paths: Tests for ``sdx_native.native_tools`` pure-Python paths and the lazy ``utils.native`` bridge.
# ==================================================================================================


def test_fnv1a64_file_counts_lines_and_bytes(tmp_path: Path):
    p = tmp_path / "t.bin"
    p.write_bytes(b"hello\nworld\n")
    h, lines, nbytes = fnv1a64_file(p)
    assert (lines, nbytes) == (2, 12)
    assert h == fnv1a64_file(p)[0]
    assert manifest_fingerprint_line(p)


def test_merge_jsonl_files_first_row_wins(tmp_path: Path):
    a, b, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    a.write_text(json.dumps({"image_path": "x.png", "v": 1}) + "\n", encoding="utf-8")
    b.write_text(
        json.dumps({"image_path": "x.png", "v": 2}) + "\n" + json.dumps({"path": "y.png"}) + "\n", encoding="utf-8"
    )
    merge_jsonl_files([a, b], out, prefer_go=False)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"image_path": "x.png", "v": 1}, {"path": "y.png"}]


def test_native_stack_status_lists_every_library():
    st = native_stack_status()
    assert all(f"lib{name}" in st for name in SHARED_LIBRARIES)


def test_utils_native_lazy_exports():
    import utils.native as n

    assert callable(n.merge_jsonl_files)
    assert callable(n.maybe_image_stats_native)
    assert "native_stack_status" in dir(n)


# ==================================================================================================
# test_plain_dict_snapshot: ``utils.runtime.to_plain_dict`` — slotted dataclasses and plain objects.
# ==================================================================================================


@dataclass(slots=True)
class _SlottedSample:
    x: int
    y: str = "hi"


def test_to_plain_dict_dataclass_slots():
    d = to_plain_dict(_SlottedSample(x=1))
    assert d == {"x": 1, "y": "hi"}


def test_to_plain_dict_plain_dict_copies():
    src = {"a": 1}
    d = to_plain_dict(src)
    assert d == src
    d["a"] = 2
    assert src["a"] == 1


def test_to_plain_dict_no_instance_dict_returns_empty():
    class Obj:
        __slots__ = ()  # noqa: PIE793 — explicit no instance dict

        def __init__(self) -> None:
            pass

    # Fallback path: no __dict__ on instance with empty slots
    o = Obj()
    assert to_plain_dict(o) == {}


def test_vit_config_snapshot():
    from vit_quality.model import ViTConfig

    d = to_plain_dict(ViTConfig())
    assert d["model_name"] == "vit_base_patch16_224"
    assert d["epochs"] == 5


def test_train_config_slotted_snapshot():
    from config.train_config import TrainConfig

    c = TrainConfig()
    d = to_plain_dict(c)
    assert d["model_name"] == c.model_name
    assert "global_batch_size" in d
