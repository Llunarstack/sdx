"""
**Reflector** — Act–Reflect–Think–Act (VisionCreator-R1 style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from research.agi_image.schemas.agent_messages import VerificationVerdict


@dataclass(slots=True)
class ReflectionOutcome:
    accepted: bool
    verdict: VerificationVerdict
    prompt_patch: str = ""
    negative_patch: str = ""


def _suggest_from_metrics(metrics: Dict[str, float], *, expected_text: str = "") -> tuple[str, str]:
    """Heuristic Think step: map failed metrics → prompt/negative patches."""
    prompt_bits: list[str] = []
    neg_bits: list[str] = []
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    sharp = float(metrics.get("sharpness", 0.0) or 0.0)

    if clip < 0.22:
        prompt_bits.append("exact subject match to description, clear focal subject")
    if sharp < 120.0:
        prompt_bits.append(" tack sharp, crisp micro-detail")
        neg_bits.append("blurry, soft focus, motion blur, plastic skin, waxy face")
    if comp < 0.55:
        prompt_bits.append("professional photography, balanced exposure, natural skin texture")
        neg_bits.append("low quality, artifacts, oversaturated, ai generated look, oversmoothed")
    if expected_text and float(metrics.get("ocr_match", 1.0) or 1.0) < 0.7:
        prompt_bits.append(f'legible text reading "{expected_text}"')
        neg_bits.append("misspelled text, garbled typography")

    return ", ".join(prompt_bits), ", ".join(neg_bits)


def reflect_on_result(
    metrics: Dict[str, float],
    *,
    iteration: int,
    min_composite: float = 0.62,
    min_clip: float = 0.22,
    expected_text: str = "",
) -> ReflectionOutcome:
    """
    Reflect on verify metrics; return acceptance + corrective patches.
    """
    comp = float(metrics.get("composite", 0.0) or 0.0)
    clip = float(metrics.get("clip", 0.0) or 0.0)
    accepted = comp >= float(min_composite) and clip >= float(min_clip)
    prompt_patch, negative_patch = ("", "")
    suggestion = ""
    if not accepted:
        prompt_patch, negative_patch = _suggest_from_metrics(metrics, expected_text=expected_text)
        suggestion = prompt_patch or "improve adherence and sharpness"

    verdict = VerificationVerdict(
        source="heuristic",
        iteration=int(iteration),
        acceptance=accepted,
        summary=f"composite={comp:.3f} clip={clip:.3f}",
        suggestion=suggestion or None,
        metrics={k: float(v) for k, v in metrics.items()},
    )
    return ReflectionOutcome(
        accepted=accepted,
        verdict=verdict,
        prompt_patch=prompt_patch,
        negative_patch=negative_patch,
    )


def reflect_on_result_llm(
    metrics: Dict[str, float],
    *,
    iteration: int,
    prompt: str,
    qwen_path: str,
    device: str = "cuda",
    min_composite: float = 0.62,
    min_clip: float = 0.22,
    expected_text: str = "",
) -> ReflectionOutcome:
    """
    LLM-backed reflect step (Qwen or other HF causal LM). Falls back to heuristics on failure.
    """
    base = reflect_on_result(
        metrics,
        iteration=iteration,
        min_composite=min_composite,
        min_clip=min_clip,
        expected_text=expected_text,
    )
    if base.accepted or not str(qwen_path or "").strip():
        return base
    try:
        from utils.analysis.llm_client import load_qwen_causal_lm

        tok, model = load_qwen_causal_lm(str(qwen_path), device=device)
        messages = [
            {
                "role": "system",
                "content": (
                    "You improve image-generation prompts after a failed verification. "
                    "Reply with two lines only: PROMPT_PATCH: ... then NEGATIVE_PATCH: ..."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original prompt: {prompt}\n"
                    f"Metrics: {metrics}\n"
                    f"Iteration: {iteration}\n"
                    "Suggest concise prompt and negative patches."
                ),
            },
        ]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"Prompt: {prompt}\nMetrics: {metrics}"
        inputs = tok(text, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        prompt_patch = ""
        negative_patch = ""
        for line in gen.splitlines():
            low = line.strip().lower()
            if low.startswith("prompt_patch:"):
                prompt_patch = line.split(":", 1)[-1].strip()
            elif low.startswith("negative_patch:"):
                negative_patch = line.split(":", 1)[-1].strip()
        if not prompt_patch and not negative_patch:
            return base
        verdict = VerificationVerdict(
            source="llm",
            iteration=int(iteration),
            acceptance=False,
            summary=base.verdict.summary,
            suggestion=prompt_patch or base.verdict.suggestion,
            metrics={k: float(v) for k, v in metrics.items()},
        )
        return ReflectionOutcome(
            accepted=False,
            verdict=verdict,
            prompt_patch=prompt_patch or base.prompt_patch,
            negative_patch=negative_patch or base.negative_patch,
        )
    except Exception:
        return base


__all__ = ["ReflectionOutcome", "reflect_on_result", "reflect_on_result_llm"]
