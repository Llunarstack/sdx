"""
Load a small JSON prompt pack for repeatable sampling / regression checks.

Schema: a JSON array of strings or of objects with a ``"prompt"`` string field.
Optional ``"id"`` on objects is preserved for logging (see ``load_eval_prompt_records``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPromptRecord:
    """One row from a JSON eval pack after normalization."""

    id: str
    prompt: str


def load_eval_prompt_records(path: Path | str) -> list[EvalPromptRecord]:
    """Parse *path* and return records with ``id`` and ``prompt`` set."""
    p = Path(path)
    raw: Any = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("eval prompt pack root must be a JSON array")
    out: list[EvalPromptRecord] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(EvalPromptRecord(id=f"p{i}", prompt=text))
        elif isinstance(item, dict):
            pr = item.get("prompt")
            if not isinstance(pr, str) or not pr.strip():
                raise ValueError(f"item {i}: missing non-empty string 'prompt'")
            eid = item.get("id")
            rid = eid.strip() if isinstance(eid, str) and eid.strip() else f"p{i}"
            out.append(EvalPromptRecord(id=rid, prompt=pr.strip()))
        else:
            raise ValueError(f"item {i}: expected str or object, got {type(item).__name__}")
    if not out:
        raise ValueError("eval prompt pack is empty")
    return out


def load_eval_prompts(path: Path | str) -> list[str]:
    """Return prompts only (same order as ``load_eval_prompt_records``)."""
    return [r.prompt for r in load_eval_prompt_records(path)]
