"""
Optional **prompt expansion** before generation (LLM or heuristic fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class ExpandConfig:
    llm_model_path: str = ""
    device: str = "cuda"
    max_new_tokens: int = 128
    use_heuristic_fallback: bool = True


_HEURISTIC_SUFFIX = ", highly detailed, professional composition, natural lighting, sharp focus"


def expand_prompt_heuristic(prompt: str) -> str:
    """Cheap expansion without LLM."""
    p = (prompt or "").strip()
    if not p:
        return p
    low = p.lower()
    extras = []
    if "lighting" not in low:
        extras.append("natural lighting")
    if "detail" not in low and "detailed" not in low:
        extras.append("highly detailed")
    if "composition" not in low:
        extras.append("balanced composition")
    if not extras:
        return p + _HEURISTIC_SUFFIX
    return p + ", " + ", ".join(extras)


def expand_prompt(
    prompt: str,
    cfg: Optional[ExpandConfig] = None,
) -> str:
    """Expand prompt via LLM if configured, else heuristic."""
    c = cfg or ExpandConfig()
    p = (prompt or "").strip()
    if not p:
        return p
    if c.llm_model_path:
        try:
            from utils.analysis.llm_client import expand_prompt_qwen, load_qwen_causal_lm

            tok, model = load_qwen_causal_lm(c.llm_model_path, device=c.device)
            out = expand_prompt_qwen(tok, model, p, device=c.device, max_new_tokens=c.max_new_tokens)
            if out.strip():
                return out.strip()
        except Exception:
            pass
    if c.use_heuristic_fallback:
        return expand_prompt_heuristic(p)
    return p


__all__ = ["ExpandConfig", "expand_prompt", "expand_prompt_heuristic"]
