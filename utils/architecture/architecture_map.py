"""
Map **2026-era architecture themes** (flow matching, hybrid AR+DiT, RAE, distillation, …)
to **SDX modules and status** — used by docs ([`docs/ARCHITECTURE_SHIFT_2026.md`](../docs/ARCHITECTURE_SHIFT_2026.md),
[`docs/DIFFUSION_LEVERAGE_ROADMAP.md`](../docs/DIFFUSION_LEVERAGE_ROADMAP.md),
[`docs/WORKFLOW_INTEGRATION_2026.md`](../docs/WORKFLOW_INTEGRATION_2026.md))
and optional introspection / tests.

This does **not** implement external models; it documents **where** related ideas exist in-repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple


class ParityStatus(str, Enum):
    """Rough alignment with an external research/product theme."""

    IMPLEMENTED = "implemented"  # Feature exists and is usable in train/sample
    PARTIAL = "partial"  # Related knobs or subset (e.g. AR blocks, not full AR tokenizer LM)
    RESEARCH = "research"  # Documented only; no dedicated trainer
    NOT_IN_REPO = "not_in_repo"  # Not present


@dataclass(frozen=True)
class ThemeMapping:
    theme_id: str
    label: str
    status: ParityStatus
    summary: str
    repo_paths: Tuple[str, ...]
    cli_flags: Tuple[str, ...] = ()


# Canonical ordered list (single source of truth for docs/tests).
THEMES: Tuple[ThemeMapping, ...] = (
    ThemeMapping(
        "flow_matching",
        "Flow matching / rectified flow",
        ParityStatus.RESEARCH,
        "VP DDPM + Gaussian diffusion; v-pred and timestep sampling relate; full flow training not drop-in.",
        ("diffusion/", "docs/MODERN_DIFFUSION.md"),
        ("--prediction-type v", "--prediction-type x0", "--timestep-sample-mode"),
    ),
    ThemeMapping(
        "diffusion_bridges",
        "Diffusion bridges (A→B distributions)",
        ParityStatus.NOT_IN_REPO,
        "Standard path is noise→image; img2img/init paths exist in sample but not full bridge training.",
        ("sample.py", "docs/MODERN_DIFFUSION.md"),
        (),
    ),
    ThemeMapping(
        "hybrid_ar_diffusion",
        "Hybrid AR planner + diffusion decoder",
        ParityStatus.PARTIAL,
        "Block-causal DiT (num_ar_blocks) + AR/ViT bridge — not a separate billion-parameter AR image tokenizer.",
        ("models/attention.py", "models/dit_text.py", "utils/ar_dit_vit.py", "docs/AR.md"),
        ("--num-ar-blocks",),
    ),
    ThemeMapping(
        "vision_mamba_ssm",
        "Mamba / SSM vision backbones",
        ParityStatus.NOT_IN_REPO,
        "DiT uses transformer attention; no Mamba backbone in this repo.",
        (),
        (),
    ),
    ThemeMapping(
        "dmd_distillation",
        "DMD / one-step / consistency distillation",
        ParityStatus.NOT_IN_REPO,
        "No DMD trainer; see IMPROVEMENTS for distillation roadmap.",
        ("docs/IMPROVEMENTS.md",),
        (),
    ),
    ThemeMapping(
        "rae_semantic_latent",
        "RAE / semantic-first latents",
        ParityStatus.IMPLEMENTED,
        "RAE autoencoder + RAELatentBridge; REPA aligns to frozen vision encoders.",
        ("models/rae_latent_bridge.py", "train.py", "sample.py"),
        ("--autoencoder-type rae", "--repa-weight"),
    ),
    ThemeMapping(
        "repa_alignment",
        "REPA (representation alignment)",
        ParityStatus.IMPLEMENTED,
        "Auxiliary alignment to DINOv2 (or configured encoder) during training.",
        ("train.py", "docs/MODEL_STACK.md"),
        ("--repa-weight", "--repa-encoder-model"),
    ),
    ThemeMapping(
        "physical_grounding",
        "Physics-informed / physical consistency layers",
        ParityStatus.NOT_IN_REPO,
        "No built-in physics simulator; optional verifier scores (e.g. exposure) in test_time_pick.",
        ("utils/test_time_pick.py", "docs/LANDSCAPE_2026.md"),
        ("--pick-best combo_exposure",),
    ),
    ThemeMapping(
        "rag_prompt_grounding",
        "RAG-style facts in prompt",
        ParityStatus.IMPLEMENTED,
        "merge_facts_into_prompt and JSONL loaders — no retrieval implementation.",
        ("utils/rag_prompt.py",),
        (),
    ),
    ThemeMapping(
        "orchestration_pipeline",
        "Multi-stage designer→verifier pipeline",
        ParityStatus.IMPLEMENTED,
        "orchestrate_pipeline.py, pick-best, orchestration roles.",
        ("scripts/tools/ops/orchestrate_pipeline.py", "utils/generation/orchestration.py"),
        ("--num", "--pick-best"),
    ),
    ThemeMapping(
        "resolution_buckets",
        "Multi-resolution / aspect buckets (training)",
        ParityStatus.IMPLEMENTED,
        "train.py --resolution-buckets; single-GPU; no val-split with buckets.",
        ("data/t2i_dataset.py", "data/bucket_batch_sampler.py", "train.py"),
        ("--resolution-buckets",),
    ),
    # --- Workflow / industry commentary (docs/WORKFLOW_INTEGRATION_2026.md) ---
    ThemeMapping(
        "workflow_structural_coherence",
        "Multi-view / structural coherence (industry stacks)",
        ParityStatus.NOT_IN_REPO,
        "No built-in 3D/pose engine; use data, control paths, character_lock, resolution tooling.",
        ("docs/WORKFLOW_INTEGRATION_2026.md", "utils/character_lock.py"),
        (),
    ),
    ThemeMapping(
        "llada_discrete_diffusion_text",
        "Discrete diffusion on text / unified LM+image (LLaDA-class ideas)",
        ParityStatus.NOT_IN_REPO,
        "SDX: T5 (AR text encoding) + DiT diffusion — not diffusion over discrete text tokens.",
        ("docs/WORKFLOW_INTEGRATION_2026.md", "docs/MODERN_DIFFUSION.md"),
        (),
    ),
    ThemeMapping(
        "test_time_inference_scaling",
        "Test-time compute / critique-style inference (industry narratives)",
        ParityStatus.PARTIAL,
        "Refinement pass, pick-best, orchestrate_pipeline — explicit scoring, not proprietary latent self-critique.",
        ("sample.py", "utils/quality/test_time_pick.py", "scripts/tools/ops/orchestrate_pipeline.py"),
        ("--num", "--pick-best"),
    ),
    ThemeMapping(
        "live_grounding_web",
        "Live web grounding during generation",
        ParityStatus.NOT_IN_REPO,
        "rag_prompt merges user-supplied facts; no built-in web retrieval in core train/sample.",
        ("utils/rag_prompt.py", "docs/WORKFLOW_INTEGRATION_2026.md"),
        (),
    ),
    ThemeMapping(
        "diffusion_leverage_roadmap",
        "Prioritized diffusion model upgrades (compound levers)",
        ParityStatus.RESEARCH,
        "Narrative doc: data/latent/objective/conditioning/inference/alignment sequencing; maps ideas to train.py, dit_text, diffusion/, RAE/REPA.",
        ("docs/DIFFUSION_LEVERAGE_ROADMAP.md", "docs/MODERN_DIFFUSION.md"),
        (),
    ),
    ThemeMapping(
        "auxiliary_structure_supervision",
        "Auxiliary depth/edge/segmentation heads on denoiser",
        ParityStatus.NOT_IN_REPO,
        "Structure losses for layout/comics not wired; would extend DiT outputs or intermediate features.",
        ("models/dit_text.py", "models/enhanced_dit.py", "docs/DIFFUSION_LEVERAGE_ROADMAP.md"),
        (),
    ),
    ThemeMapping(
        "reference_image_adapter_conditioning",
        "Reference-image / IP-Adapter-style conditioning for identity lock",
        ParityStatus.PARTIAL,
        "CLIP vision -> ReferenceTokenProjector -> extra cross-attn tokens in DiT_Text; optional trained --reference-adapter-pt. Also --init-image, --post-reference-image, character sheet.",
        ("models/dit_text.py", "models/reference_token_projection.py", "sample.py", "utils/generation/clip_reference_embed.py"),
        (
            "--reference-image",
            "--reference-strength",
            "--reference-tokens",
            "--reference-adapter-pt",
            "--init-image",
            "--post-reference-image",
            "--character-sheet",
        ),
    ),
    ThemeMapping(
        "self_attention_guidance_blur",
        "Blur-based self-attention guidance (extra forward on blurred latent)",
        ParityStatus.IMPLEMENTED,
        "sample_loop: pred += sag_scale * (pred - pred(blur(x))); ~2× forwards when sag_scale and sag_blur_sigma > 0.",
        ("diffusion/gaussian_diffusion.py", "diffusion/sampling_utils.py", "sample.py"),
        ("--sag-blur-sigma", "--sag-scale"),
    ),
)


def theme_by_id(theme_id: str) -> Optional[ThemeMapping]:
    """Return theme by ``theme_id`` or None."""
    tid = (theme_id or "").strip().lower()
    for t in THEMES:
        if t.theme_id == tid:
            return t
    return None


def iter_themes() -> Iterator[ThemeMapping]:
    """Yield all theme mappings in stable order."""
    yield from THEMES


def themes_as_dict() -> List[Dict[str, object]]:
    """JSON-friendly list for tooling."""
    out: List[Dict[str, object]] = []
    for t in THEMES:
        out.append(
            {
                "theme_id": t.theme_id,
                "label": t.label,
                "status": t.status.value,
                "summary": t.summary,
                "repo_paths": list(t.repo_paths),
                "cli_flags": list(t.cli_flags),
            }
        )
    return out


def summary_table_md() -> str:
    """Minimal Markdown table for pasting into docs or logs."""
    lines = [
        "| Theme | Status | SDX pointers |",
        "| :--- | :--- | :--- |",
    ]
    for t in THEMES:
        ptr = ", ".join(t.repo_paths[:2]) if t.repo_paths else "—"
        lines.append(f"| **{t.label}** | `{t.status.value}` | {ptr} |")
    return "\n".join(lines)
