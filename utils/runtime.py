"""Runtime helpers: UTF-8 console stdio on Windows, fast JSON (orjson when installed), plain-dict snapshots and an optional cProfile wrapper."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def configure_stdio_for_console() -> None:
    """
    Avoid UnicodeEncodeError on Windows terminals using legacy encodings (e.g. cp1252).

    Sets UTF-8 with ``errors="replace"`` so ``--help`` and logging stay robust when
    output includes non-ASCII text (typographic arrows, emoji, etc.).
    """
    if os.name != "nt":
        return
    try:
        stdout_rc = getattr(sys.stdout, "reconfigure", None)
        if callable(stdout_rc):
            stdout_rc(encoding="utf-8", errors="replace")
        stderr_rc = getattr(sys.stderr, "reconfigure", None)
        if callable(stderr_rc):
            stderr_rc(encoding="utf-8", errors="replace")
    except Exception:
        pass


try:
    import orjson
except ImportError:
    orjson = None  # type: ignore[misc, assignment]


def json_loads(data: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data.encode("utf-8") if isinstance(data, str) else data)
    return json.loads(data.decode("utf-8") if isinstance(data, bytes) else data)


def json_dumps(
    obj: Any,
    *,
    indent: int | bool | None = None,
    sort_keys: bool = False,
    ensure_ascii: bool = True,
    default: Callable[..., Any] | None = None,
) -> str:
    """Serialize like ``json.dumps``; uses ``orjson`` for ``ensure_ascii=False`` with no or 2-space indent."""
    ind = None if indent in (None, False) else 2 if indent is True else int(indent)
    if default is not None or ind not in (None, 2) or ensure_ascii or orjson is None:
        return json.dumps(obj, indent=ind, sort_keys=sort_keys, ensure_ascii=ensure_ascii, default=default)
    option = (orjson.OPT_SORT_KEYS if sort_keys else 0) | (orjson.OPT_INDENT_2 if ind == 2 else 0)
    return orjson.dumps(obj, option=option).decode("utf-8")


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """
    Build a JSON-serialisation-friendly mapping.

    Order: ``dict`` shallow copy → dataclass ``asdict`` → ``__dict__`` copy → ``vars``;
    on total failure returns ``{}`` (manifest / checkpoint writers must not crash).
    """
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    od = getattr(obj, "__dict__", None)
    if isinstance(od, dict):
        return dict(od)
    try:
        return dict(vars(obj))
    except TypeError:
        return {}


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    out_path: str
    sort_key: str
    top_n: int


def consume_profile_args(argv: Sequence[str]) -> tuple[list[str], ProfileConfig | None]:
    """Remove ``--profile-out`` / ``--profile-sort`` / ``--profile-top`` so the main parser stays unchanged."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-out", default="", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "tottime", "ncalls", "name"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--profile-top", type=int, default=80, metavar="N", help=argparse.SUPPRESS)
    if len(argv) < 2:
        return list(argv), None
    ns, rest = parser.parse_known_args(argv[1:])
    out = str(ns.profile_out or "").strip()
    if not out:
        return list(argv), None
    return [argv[0]] + rest, ProfileConfig(out_path=out, sort_key=str(ns.profile_sort), top_n=max(1, ns.profile_top))


def run_with_cprofile(entry: Callable[[], None], cfg: ProfileConfig) -> None:
    out = Path(cfg.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    try:
        entry()
    finally:
        prof.disable()
        prof.dump_stats(str(out))
        summary_path = Path(str(out) + ".txt")
        with summary_path.open("w", encoding="utf-8") as fh:
            stats = pstats.Stats(prof, stream=fh)
            stats.sort_stats(cfg.sort_key)
            stats.print_stats(cfg.top_n)
        print(f"cProfile: wrote {out} and {summary_path}", file=sys.stderr)
