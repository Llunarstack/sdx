"""Optional **RAG-style grounding** for prompts (LANDSCAPE §5, IMPROVEMENTS alignment).

Models only see frozen encoder inputs. To reflect **user-supplied facts** (retrieved docs,
product specs, “today’s” copy), merge external text into the prompt **before** encoding.

This module does **not** call the web or any API — it only formats strings. Wire your own
retrieval (vector DB, HTTP, filesystem) and pass ``facts`` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union


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


def _append_if_text(out: List[str], seen: set, value: Any, *, max_item_chars: int = 280) -> None:
    s = str(value or "").strip()
    if not s:
        return
    if len(s) > max_item_chars:
        s = s[: max_item_chars - 3].rstrip() + "..."
    k = s.lower()
    if k in seen:
        return
    seen.add(k)
    out.append(s)


def _extract_text_lines(obj: Any, out: List[str], seen: set, *, max_item_chars: int = 280) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        _append_if_text(out, seen, obj, max_item_chars=max_item_chars)
        return
    if isinstance(obj, dict):
        # Preferred keys commonly seen in search outputs.
        for k in ("text", "snippet", "summary", "title", "caption", "description", "content", "query"):
            if k in obj:
                _append_if_text(out, seen, obj.get(k), max_item_chars=max_item_chars)
        # Also recursively walk nested values.
        for v in obj.values():
            if isinstance(v, (dict, list, tuple, str)):
                _extract_text_lines(v, out, seen, max_item_chars=max_item_chars)
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _extract_text_lines(x, out, seen, max_item_chars=max_item_chars)


def facts_from_gen_searcher_payload(
    payload: Any,
    *,
    max_entries: int = 24,
    max_item_chars: int = 280,
) -> List[str]:
    """
    Extract text facts from Gen-Searcher-like outputs (JSON object/list).

    Works with common fields from agentic-search style outputs, while remaining
    robust to schema drift by recursively scanning nested text-bearing fields.
    """
    out: List[str] = []
    seen: set = set()
    if payload is None:
        return out

    if isinstance(payload, dict):
        preferred_keys = (
            "reasoning_summary",
            "search_summary",
            "final_answer",
            "retrieved_facts",
            "evidence",
            "web_evidence",
            "search_results",
            "image_results",
            "references",
            "sources",
            "results",
        )
        for k in preferred_keys:
            if k in payload:
                _extract_text_lines(payload.get(k), out, seen, max_item_chars=max_item_chars)
                if len(out) >= max_entries:
                    return out[:max_entries]
    _extract_text_lines(payload, out, seen, max_item_chars=max_item_chars)
    return out[:max_entries]


def load_facts_from_gen_searcher_json(
    path: Union[str, Path],
    *,
    max_entries: int = 24,
    max_item_chars: int = 280,
) -> List[str]:
    """
    Load Gen-Searcher-style facts from either .json or .jsonl file.

    - JSON: object or list
    - JSONL: one object per line (facts extracted from each row)
    """
    p = Path(path)
    if not p.is_file():
        return []
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        out: List[str] = []
        seen: set = set()
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for fact in facts_from_gen_searcher_payload(row, max_entries=max_entries, max_item_chars=max_item_chars):
                    k = fact.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(fact)
                    if len(out) >= max_entries:
                        return out
        return out

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return facts_from_gen_searcher_payload(payload, max_entries=max_entries, max_item_chars=max_item_chars)
