"""Bridge SDX frontier image analyzers with video frontier compiler."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

__all__ = ["bridge_shot_prompt", "list_video_frontier_modules", "unified_frontier_augment"]


def list_video_frontier_modules() -> List[Dict[str, str]]:
    from pipelines.video.frontier_compiler import list_frontier_modules

    return list_frontier_modules()


def bridge_shot_prompt(
    prompt: str,
    *,
    tension: float = 0.5,
    num_steps: int = 28,
) -> Dict[str, Any]:
    """Augment a single keyframe prompt using SDX + video frontier."""
    from pipelines.video.causal_events import apply_causal_ripples, parse_causal_rules
    from pipelines.video.mise_en_scene import compose_shot_framing, parse_mise_config

    from frontier.causality.physical_plausibility import PhysicalPlausibilityScanner
    from frontier.engine import analyze_prompt
    from frontier.narrative.moment import TemporalMomentAnalyzer
    from frontier.narrative.tension_field import build_tension_field

    class _Shot:
        def __init__(self, p: str) -> None:
            self.id = "bridge"
            self.prompt = p
            self.shot_type = "medium"
            self.camera = ""

    plan = analyze_prompt(prompt, num_steps=num_steps)
    tension_plan = build_tension_field(num_steps).plan(tension)
    moment = TemporalMomentAnalyzer(num_steps=num_steps).analyze(prompt)
    plaus = PhysicalPlausibilityScanner().scan(prompt)
    ripple = apply_causal_ripples([_Shot(prompt)], parse_causal_rules({}), use_builtins=True)
    mise = compose_shot_framing(_Shot(prompt), config=parse_mise_config({"enabled": True}))

    frags: List[str] = list(
        plan.augmented_prompt.replace(prompt, "").strip(", ").split(", ") if plan.augmented_prompt != prompt else []
    )
    frags.extend(tension_plan.prompt_fragments)
    frags.extend(moment.prompt_fragments)
    if mise:
        frags.append(mise.prompt_suffix)
    if ripple:
        frags.append(ripple[0].prompt_suffix)

    negatives: List[str] = []
    if plan.echo_negative:
        negatives.append(plan.echo_negative)
    for flag in plaus:
        if flag.severity >= 0.5:
            negatives.append(f"missing {flag.missing}")

    augmented = prompt
    clean_frags = [f.strip() for f in frags if f.strip()]
    if clean_frags:
        augmented = f"{prompt}, {', '.join(dict.fromkeys(clean_frags))}"

    return {
        "augmented_prompt": augmented,
        "negative_suffix": ", ".join(dict.fromkeys(negatives)),
        "tension": tension,
        "step_emphasis": list(tension_plan.step_emphasis),
        "cfg_boost": tension_plan.cfg_boost,
        "moment_phase": moment.phase.value,
        "risk_score": plan.risk_score,
        "plausibility_flags": [{"trigger": f.trigger, "missing": f.missing, "severity": f.severity} for f in plaus],
    }


def unified_frontier_augment(
    data: Mapping[str, Any],
    shots: List[Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    from pipelines.video.frontier_compiler import compile_frontier_layers

    result = compile_frontier_layers(data, shots, **kwargs)
    return {
        "metadata": result.metadata,
        "issues": result.issues,
        "global_edit": result.global_edit,
        "enrichments": {
            k: {
                "prompt_suffix": v.prompt_suffix,
                "negative_suffix": v.negative_suffix,
                "duration_delta": v.duration_delta,
                "metadata": v.metadata,
            }
            for k, v in result.enrichments.items()
        },
    }
