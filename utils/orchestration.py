"""Multi-stage generation **roles** (industry pattern: Designer / Verifier / Reasoner).

Production systems increasingly compose **several** models or passes instead of one
monolithic forward. SDX’s default path is still **single-stack** diffusion
(``sample.py`` + DiT + VAE), with **optional** test-time selection (``--pick-best``).

This module gives **stable names** and hints for documentation and future wiring.
See **docs/LANDSCAPE_2026.md** for industry context and **docs/IMPROVEMENTS.md** §12
for roadmap items.

Example (introspection only)::

    from utils.orchestration import pipeline_roles
    for r in pipeline_roles():
        print(r.name, "->", r.sdx_module_hint)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineRole:
    """One stage in a multi-expert image generation pipeline."""

    name: str
    description: str
    sdx_module_hint: str


DESIGNER = PipelineRole(
    name="designer",
    description="Core layout, composition, and latent diffusion (DiT + scheduler + VAE decode).",
    sdx_module_hint="sample.py, models/dit_text.py, diffusion/gaussian_diffusion.py",
)

VERIFIER = PipelineRole(
    name="verifier",
    description="Quality and consistency checks: anatomy, sharpness, text OCR match; optional refine.",
    sdx_module_hint="utils/test_time_pick.py, ViT/, sample.py refinement flags",
)

REASONER = PipelineRole(
    name="reasoner",
    description="Instruction understanding and grounding: T5 (+ optional CLIP fusion), optional LLM expansion.",
    sdx_module_hint="utils/text_encoder_bundle.py, utils/llm_client.py, JSONL region_captions",
)


def pipeline_roles() -> tuple[PipelineRole, ...]:
    """Return the canonical ordered tuple of pipeline roles."""
    return (DESIGNER, VERIFIER, REASONER)


def sample_cli_hint(
    *,
    num_candidates: int = 4,
    pick_metric: str = "combo",
    out_path: str = "out.png",
) -> str:
    """
    Return a minimal ``sample.py`` invocation string for multi-stage (Designer + Verifier) flows.

    See **scripts/tools/orchestrate_pipeline.py** for a ready-made wrapper.
    """
    return (
        f'python sample.py --ckpt CKPT --prompt "..." --num {num_candidates} '
        f'--pick-best {pick_metric} --out {out_path}'
    )
