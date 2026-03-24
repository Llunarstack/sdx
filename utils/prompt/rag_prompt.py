"""Optional **RAG-style grounding** for prompts (LANDSCAPE §5, IMPROVEMENTS alignment).

Models only see frozen encoder inputs. To reflect **user-supplied facts** (retrieved docs,
product specs, “today’s” copy), merge external text into the prompt **before** encoding.

This module does **not** call the web or any API — it only formats strings. Wire your own
retrieval (vector DB, HTTP, filesystem) and pass ``facts`` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


def merge_facts_into_prompt(
    user_prompt: str,
    facts: Sequence[str],
    *,
    prefix: str = "Context (user-supplied facts, treat as authoritative):\n",
    sep: str = "\n- ",
    suffix: str = "\n\nTask:\n",
    max_chars: int = 6000,
) -> str:
    """
    Prepend structured facts so the diffusion prompt conditions on explicit grounding.

    Truncates joined facts to ``max_chars`` (keeps the start). Empty ``facts`` returns
    ``user_prompt`` unchanged.
    """
    user_prompt = (user_prompt or "").strip()
    lines = [str(f).strip() for f in facts if f is not None and str(f).strip()]
    if not lines:
        return user_prompt
    block = prefix + sep + sep.join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 3] + "..."
    return (block + suffix + user_prompt).strip()


def load_facts_from_jsonl(
    path: Union[str, Path],
    *,
    text_key: str = "text",
    max_entries: int = 32,
) -> List[str]:
    """
    Load up to ``max_entries`` string fields from a JSONL file (one JSON object per line).

    Each line should be like ``{\"text\": \"...\"}`` or use ``text_key`` for the fact body.
    Lines without ``text_key`` are skipped.
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: List[str] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get(text_key)
            if t is None:
                continue
            out.append(str(t).strip())
            if len(out) >= max_entries:
                break
    return out


def facts_from_mapping(items: Iterable[Dict[str, Any]], *, text_key: str = "text") -> List[str]:
    """Extract ``text_key`` from each dict in ``items``."""
    return [str(d[text_key]).strip() for d in items if text_key in d and str(d.get(text_key, "")).strip()]
