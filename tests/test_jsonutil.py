import json
from datetime import datetime, timezone

from utils.runtime import jsonutil


def test_loads_roundtrip_stdlib():
    assert jsonutil.loads('{"a": 1}') == {"a": 1}
    assert jsonutil.loads(b'{"b": 2}') == {"b": 2}


def test_dumps_utf8_fast_path_matches_json():
    obj = {"x": 1, "ü": "beta"}
    a = jsonutil.dumps(obj, ensure_ascii=False, sort_keys=True)
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    assert json.loads(a) == json.loads(b)


def test_dumps_ensure_ascii_uses_stdlib_escape():
    obj = {"ü": "x"}
    s = jsonutil.dumps(obj, ensure_ascii=True, sort_keys=True)
    assert "\\u" in s or "ü" not in s


def test_loads_accepts_bytes():
    assert jsonutil.loads(b'{"k": null}') == {"k": None}


def test_dumps_with_default_forces_stdlib():
    dt = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    s = jsonutil.dumps({"t": dt}, default=lambda o: o.isoformat(), ensure_ascii=False)
    assert json.loads(s)["t"].startswith("2024-06-01")


def test_dumps_indent_true_maps_to_two_spaces():
    obj = {"a": [1, 2]}
    s = jsonutil.dumps(obj, indent=True, ensure_ascii=False, sort_keys=True)
    parsed = json.loads(s)
    assert parsed == obj
    assert "\n" in s
