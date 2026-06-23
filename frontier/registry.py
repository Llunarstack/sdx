"""
Research-backed idea catalog for frontier experiments.

Each entry links to a paper/project and notes implementation status in SDX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

Status = Literal["implemented", "partial", "planned", "research"]


@dataclass(frozen=True)
class ResearchIdea:
    id: str
    title: str
    source: str
    url: str
    status: Status
    module: str
    summary: str


IDEAS: List[ResearchIdea] = [
    ResearchIdea(
        id="regional_cfg",
        title="Training-free regional prompting (DiT)",
        source="Regional-Prompting-FLUX / InstantX",
        url="https://arxiv.org/html/2411.02395",
        status="implemented",
        module="utils/generation/regional_box_prompting.py",
        summary="Per-box prompts + masks blended during CFG; mask_inject_steps + base_ratio.",
    ),
    ResearchIdea(
        id="box_sketch",
        title="Draw + describe per region",
        source="Ideogram / sketch-conditioned layout",
        url="https://ideogram.ai",
        status="implemented",
        module="utils/generation/regional_box_sketch.py",
        summary="Vector strokes or sketch PNG per box with text prompt.",
    ),
    ResearchIdea(
        id="omost_canvas",
        title="Omost Canvas → layout DSL",
        source="lllyasviel/Omost",
        url="https://github.com/lllyasviel/Omost",
        status="implemented",
        module="frontier/layout/omost_canvas.py",
        summary="LLM-friendly Canvas API compiles to box-layout JSON.",
    ),
    ResearchIdea(
        id="consist_compose",
        title="Coordinate-bound prompts (LELG)",
        source="ConsistCompose CVPR 2026",
        url="https://openaccess.thecvf.com/content/CVPR2026/html/Shi_ConsistCompose_Unified_Multimodal_Layout_Control_for_Image_Composition_CVPR_2026_paper.html",
        status="implemented",
        module="frontier/layout/coordinate_bind.py",
        summary="Embed bbox tokens in prompts for coordinate-aware CFG.",
    ),
    ResearchIdea(
        id="lamic_schedule",
        title="LAMIC region-modulated fusion schedule",
        source="LAMIC AAAI 2026",
        url="https://arxiv.org/html/2508.00477",
        status="implemented",
        module="frontier/layout/lamic_schedule.py",
        summary="Early-step region isolation, late-step cross-region fusion.",
    ),
    ResearchIdea(
        id="lamic_metrics",
        title="Layout metrics IN-R / FI-R / BG-S",
        source="LAMIC AAAI 2026",
        url="https://ojs.aaai.org/index.php/AAAI/article/view/37311",
        status="implemented",
        module="frontier/layout/layout_metrics.py",
        summary="Inclusion and fill ratios for box-layout QA.",
    ),
    ResearchIdea(
        id="dynamic_cfg",
        title="Dynamic CFG via latent feedback",
        source="Google Imagen 3 research",
        url="https://arxiv.org/html/2509.16131",
        status="implemented",
        module="frontier/guidance/dynamic_cfg.py",
        summary="Greedy per-step CFG pick from cheap latent heuristics.",
    ),
    ResearchIdea(
        id="cfg_interval",
        title="CFG guidance intervals",
        source="Kynkäänniemi et al. / CFG scheduler analysis",
        url="https://ar5iv.labs.arxiv.org/html/2404.13040",
        status="implemented",
        module="frontier/guidance/guidance_interval.py",
        summary="Skip CFG early/late; monotonic cosine ramp mid-schedule.",
    ),
    ResearchIdea(
        id="cross_attn_layout",
        title="Cross-attention layout guidance",
        source="BoxDiff / Attend-and-Excite / Dense Diffusion",
        url="https://arxiv.org/html/2304.03373",
        status="partial",
        module="frontier/attention/layout_plan.py",
        summary="Training-free plan: which steps to enforce box attention (hook point).",
    ),
    ResearchIdea(
        id="multi_ref_region",
        title="Per-region identity / reference image",
        source="Regional-Prompting-FLUX + PULID",
        url="https://github.com/instantX-research/Regional-Prompting-FLUX",
        status="implemented",
        module="frontier/compose/multi_reference.py",
        summary="Attach reference image per box for identity-locked regions.",
    ),
    ResearchIdea(
        id="metapoint",
        title="MetaPoint spatial tokens",
        source="MetaPoint 2026",
        url="https://arxiv.org/html/2606.05031",
        status="planned",
        module="frontier/layout/",
        summary="Coordinate tokens for move/add/remove/edit — needs DiT token hook.",
    ),
    ResearchIdea(
        id="dense_diffusion",
        title="Dense Diffusion attention masking",
        source="Naver Dense Diffusion",
        url="https://github.com/naver-ai/DenseDiffusion",
        status="planned",
        module="models/",
        summary="Modify q@k scores per region inside cross-attn blocks.",
    ),
    ResearchIdea(
        id="cads",
        title="CADS condition annealing",
        source="CADS / holy grail",
        url="https://github.com/huggingface/diffusers",
        status="partial",
        module="diffusion/sampling/",
        summary="Already in sampling helpers; extend to per-region condition noise.",
    ),
    ResearchIdea(
        id="self_refine_loop",
        title="Generate → VLM critique → re-prompt",
        source="Agentic T2I 2025–2026",
        url="https://arxiv.org/html/2606.05031",
        status="partial",
        module="innovations/agentic/",
        summary="Wire frontier layout metrics into iterative refinement loop.",
    ),
]


def list_ideas(*, status: Status | None = None) -> List[ResearchIdea]:
    if status is None:
        return list(IDEAS)
    return [i for i in IDEAS if i.status == status]


def idea_by_id(idea_id: str) -> ResearchIdea:
    key = (idea_id or "").strip().lower()
    for i in IDEAS:
        if i.id == key:
            return i
    raise KeyError(f"unknown idea {idea_id!r}")


def ideas_summary_markdown() -> str:
    lines = ["| ID | Status | Module |", "|----|--------|--------|"]
    for i in IDEAS:
        lines.append(f"| `{i.id}` | {i.status} | `{i.module}` |")
    return "\n".join(lines)


__all__ = [
    "IDEAS",
    "ResearchIdea",
    "idea_by_id",
    "ideas_summary_markdown",
    "list_ideas",
]
