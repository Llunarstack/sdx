"""
Faster JSON when ``orjson`` is installed (``pip install -r requirements-perf.txt``).

Falls back to the standard library so the repo stays usable without optional deps.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Union

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore[misc, assignment]

__all__ = ["dumps", "loads"]


def loads(data: Union[str, bytes]) -> Any:
    if orjson is not None:
        if isinstance(data, str):
            return orjson.loads(data.encode("utf-8"))
        return orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def _stdlib_indent(indent: Optional[Union[int, bool]]) -> Optional[int]:
    if indent in (None, False):
        return None
    if indent is True:
        return 2
    return int(indent)


def dumps(
    obj: Any,
    *,
    indent: Optional[Union[int, bool]] = None,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    default: Optional[Callable[..., Any]] = None,
) -> str:
    """
    Serialize like ``json.dumps``.

    When ``ensure_ascii=False`` and indent is unset or 2-space, uses ``orjson`` if installed.
    """
    if default is not None:
        return json.dumps(
            obj,
            indent=_stdlib_indent(indent),
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            default=default,
        )
    ind = _stdlib_indent(indent)
    if ind not in (None, 2):
        return json.dumps(obj, indent=ind, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    if ensure_ascii:
        return json.dumps(obj, indent=ind, sort_keys=sort_keys, ensure_ascii=True)
    if orjson is None:
        return json.dumps(obj, indent=ind, sort_keys=sort_keys, ensure_ascii=False)
    option = 0
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if ind == 2:
        option |= orjson.OPT_INDENT_2
    return orjson.dumps(obj, option=option).decode("utf-8")
