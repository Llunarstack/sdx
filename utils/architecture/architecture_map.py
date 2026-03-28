"""
Map **2026-era architecture themes** (flow matching, hybrid AR+DiT, RAE, distillation, …)
to **SDX modules and status** — used by docs ([`docs/LANDSCAPE_2026.md`](../docs/LANDSCAPE_2026.md),
[`docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md`](../docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md),
[`docs/BLUEPRINTS.md`](../docs/BLUEPRINTS.md),
[`docs/DIFFUSION_LEVERAGE_ROADMAP.md`](../docs/DIFFUSION_LEVERAGE_ROADMAP.md))
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
        ParityStatus.PARTIAL,
        "Train: --flow-matching-training + diffusion/flow_matching.py. Sample: sample_loop(..., flow_matching_sample=True) "
        "or sample.py --flow-matching-sample / auto when checkpoint has flow_matching_training.",
        (
            "diffusion/flow_matching.py",
            "diffusion/gaussian_diffusion.py",
            "sample.py",
            "train.py",
            "docs/MODERN_DIFFUSION.md",
        ),
        ("--flow-matching-training", "--flow-matching-sample", "--flow-solver", "--prediction-type v"),
    ),
    ThemeMapping(
        "diffusion_bridges",
        "Diffusion bridges (A→B distributions)",
        ParityStatus.PARTIAL,
        "Linear latent interpolation (diffusion/latent_bridge.py) + optional VP shuffle-pair auxiliary loss in train "
        "(diffusion/bridge_training.py, --bridge-aux-weight); not a full Schrödinger-bridge trainer.",
        ("diffusion/latent_bridge.py", "diffusion/bridge_training.py", "train.py", "sample.py", "docs/MODERN_DIFFUSION.md"),
        ("--bridge-aux-weight", "--bridge-aux-lambda"),
    ),
    ThemeMapping(
        "hybrid_ar_diffusion",
        "Hybrid AR planner + diffusion decoder",
        ParityStatus.PARTIAL,
        "Block-causal DiT (num_ar_blocks) + AR/ViT bridge — not a separate billion-parameter AR image tokenizer.",
        ("models/attention.py", "models/dit_text.py", "utils/architecture/ar_dit_vit.py", "docs/AR.md"),
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
        ParityStatus.PARTIAL,
        "Same-arch teacher→student MSE on shared (x_t,t) in scripts/tools/training/train_kd_distill.py; "
        "no DMD / one-step student or ADD.",
        ("scripts/tools/training/train_kd_distill.py", "docs/IMPROVEMENTS.md"),
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
        ("utils/quality/test_time_pick.py", "docs/LANDSCAPE_2026.md"),
        ("--pick-best combo_exposure",),
    ),
    ThemeMapping(
        "rag_prompt_grounding",
        "RAG-style facts in prompt",
        ParityStatus.IMPLEMENTED,
        "merge_facts_into_prompt and JSONL loaders — no retrieval implementation.",
        ("utils/prompt/rag_prompt.py",),
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
    # --- Workflow / industry commentary (docs/LANDSCAPE_2026.md) ---
    ThemeMapping(
        "workflow_structural_coherence",
        "Multi-view / structural coherence (industry stacks)",
        ParityStatus.NOT_IN_REPO,
        "No built-in 3D/pose engine; use data, control paths, character_lock, resolution tooling.",
        ("docs/LANDSCAPE_2026.md", "utils/consistency/character_lock.py"),
        (),
    ),
    ThemeMapping(
        "llada_discrete_diffusion_text",
        "Discrete diffusion on text / unified LM+image (LLaDA-class ideas)",
        ParityStatus.NOT_IN_REPO,
        "SDX: T5 (AR text encoding) + DiT diffusion — not diffusion over discrete text tokens.",
        ("docs/LANDSCAPE_2026.md", "docs/MODERN_DIFFUSION.md"),
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
        ("utils/prompt/rag_prompt.py", "docs/LANDSCAPE_2026.md"),
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
    # --- Next-gen “super-model” pillars (design doc: docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md) ---
    ThemeMapping(
        "nextgen_semantic_geometric_dual",
        "Next-gen: semantic–geometric dual backbone (global planner + DiT)",
        ParityStatus.PARTIAL,
        "No separate Mamba architect network; dual-stage layout, block AR, --ssm-every-n mixer, and rich conditioning approximate layout-first→detail.",
        (
            "docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md",
            "sample.py",
            "utils/generation/inference_research_hooks.py",
            "models/dit_text.py",
        ),
        ("--dual-stage-layout", "--num-ar-blocks", "--ssm-every-n"),
    ),
    ThemeMapping(
        "nextgen_vlm_in_loop_critic",
        "Next-gen: in-loop VLM / discriminative correction",
        ParityStatus.PARTIAL,
        "CLIP guard, optional mid-loop CLIP CFG monitor, volatile CFG; not a full VLM gradient critic or localized rewind.",
        (
            "docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md",
            "utils/generation/clip_alignment.py",
            "diffusion/gaussian_diffusion.py",
            "sample.py",
        ),
        ("--clip-guard-threshold", "--clip-monitor-every", "--volatile-cfg-boost", "--pick-best"),
    ),
    ThemeMapping(
        "nextgen_fourier_operator_diffusion",
        "Next-gen: Fourier / neural-operator style diffusion",
        ParityStatus.PARTIAL,
        "Spectral SFP loss + inference spectral-coherence latent blend; not full NOD denoising on arbitrary grids.",
        (
            "docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md",
            "diffusion/spectral_sfp.py",
            "utils/generation/inference_research_hooks.py",
            "docs/MODERN_DIFFUSION.md",
        ),
        ("--spectral-sfp-loss", "--spectral-coherence-latent"),
    ),
    ThemeMapping(
        "nextgen_dpo_image_alignment",
        "Next-gen: direct preference optimization on images / latents",
        ParityStatus.PARTIAL,
        "Stage-2 script train_diffusion_dpo.py + DPO loss + preference JSONL/image dataset; not merged into train.py main loop.",
        (
            "docs/NEXTGEN_SUPERMODEL_ARCHITECTURE.md",
            "scripts/tools/training/train_diffusion_dpo.py",
            "utils/training/diffusion_dpo_loss.py",
            "utils/training/preference_jsonl.py",
            "utils/training/preference_image_dataset.py",
            "diffusion/gaussian_diffusion.py",
        ),
        (),
    ),
    # --- Fast-gen math blueprint (docs/BLUEPRINTS.md) ---
    ThemeMapping(
        "consistency_flow_matching_velocity",
        "Consistency / velocity self-consistency on flow fields (Consistency-FM class)",
        ParityStatus.PARTIAL,
        "Prototype rectified-flow-style velocity MSE in train (--flow-matching-training, diffusion/flow_matching.py); "
        "not Consistency-FM trajectory matching and not VP-compatible sampling without a flow sampler.",
        (
            "docs/BLUEPRINTS.md",
            "diffusion/flow_matching.py",
            "train.py",
            "diffusion/gaussian_diffusion.py",
            "docs/MODERN_DIFFUSION.md",
        ),
        ("--flow-matching-training", "--prediction-type v", "--steps"),
    ),
    ThemeMapping(
        "dual_solver_time_warping",
        "Dual-regime solvers: ε/v/x₀ mixing + log vs linear time (learned τ)",
        ParityStatus.PARTIAL,
        "Single prediction type per checkpoint; non-uniform training timesteps (logit_normal, high_noise) — not dynamic dual integrator.",
        (
            "docs/BLUEPRINTS.md",
            "diffusion/timestep_sampling.py",
            "train.py",
            "sample.py",
        ),
        ("--prediction-type", "--timestep-sample-mode"),
    ),
    ThemeMapping(
        "add_adversarial_distillation",
        "Adversarial Diffusion Distillation (ADD-class teacher→student)",
        ParityStatus.NOT_IN_REPO,
        "No adversarial distillation or dual-head discriminator student training.",
        ("docs/BLUEPRINTS.md", "docs/IMPROVEMENTS.md"),
        (),
    ),
    ThemeMapping(
        "rectified_flow_ot_coupling",
        "Rectified flow + OT pairing (Sinkhorn / Hungarian noise–data coupling)",
        ParityStatus.PARTIAL,
        "Optional mini-batch OT noise coupling (--ot-noise-pair-reg) and optional flow-matching training path "
        "(--flow-matching-training); not a full continuous-time rectified-flow ODE trainer + sampler.",
        (
            "docs/BLUEPRINTS.md",
            "utils/training/ot_noise_pairing.py",
            "diffusion/flow_matching.py",
            "train.py",
            "docs/MODERN_DIFFUSION.md",
        ),
        ("--ot-noise-pair-reg", "--ot-noise-pair-mode", "--flow-matching-training"),
    ),
    ThemeMapping(
        "speculative_cfg_denoise",
        "Speculative draft CFG (two forwards, optional blend when predictions agree)",
        ParityStatus.PARTIAL,
        "Same-backbone draft+full CFG in GaussianDiffusion.sample_loop via utils/generation/speculative_denoise.py.",
        ("utils/generation/speculative_denoise.py", "diffusion/gaussian_diffusion.py", "sample.py"),
        ("--speculative-draft-cfg-scale", "--speculative-close-thresh", "--speculative-blend"),
    ),
    # --- Prompt-accuracy blueprint (docs/BLUEPRINTS.md) ---
    ThemeMapping(
        "geometric_latent_split_blueprint",
        "Geometric–latent split (GLS): structural blueprint before texture",
        ParityStatus.PARTIAL,
        "Dual-stage layout and domain/layout priors in hooks; no dedicated depth–normal–edge transformer locked as immutable constraint.",
        (
            "docs/BLUEPRINTS.md",
            "sample.py",
            "utils/generation/inference_research_hooks.py",
        ),
        ("--dual-stage-layout", "--domain-prior-latent"),
    ),
    ThemeMapping(
        "discriminative_denoise_vlm_loop",
        "Discriminative denoising: VLM / critic in the sampling loop",
        ParityStatus.PARTIAL,
        "CLIP guard refine and volatile CFG; no frozen VLM every-k-steps with localized latent rewind / gradient reroll.",
        (
            "docs/BLUEPRINTS.md",
            "utils/generation/clip_alignment.py",
            "sample.py",
        ),
        (
            "--clip-guard-threshold",
            "--clip-monitor-every",
            "--volatile-cfg-boost",
            "--pick-best",
        ),
    ),
    ThemeMapping(
        "frequency_domain_global_coherence",
        "Frequency-domain / FNO-style global coherence (prompt-scale narrative)",
        ParityStatus.PARTIAL,
        "Spectral SFP training loss + inference ``--spectral-coherence-latent`` (FFT lowfreq blend); not full FNO denoising forward.",
        (
            "docs/BLUEPRINTS.md",
            "diffusion/spectral_sfp.py",
            "utils/generation/inference_research_hooks.py",
            "docs/MODERN_DIFFUSION.md",
        ),
        ("--spectral-sfp-loss", "--spectral-coherence-latent"),
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
