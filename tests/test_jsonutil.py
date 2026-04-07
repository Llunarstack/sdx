import json

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
