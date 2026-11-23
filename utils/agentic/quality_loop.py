"""Quality loops around sample.py: multi-candidate ensembles and composite ranking, TF-IDF prompt enrichment, self-correction and quality gates, and the auto-improve flywheel (benchmark -> mine -> DPO -> soup -> report)."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config.presets import FlywheelPlan
from config.prompts import merge_csv_unique
from PIL import Image

from utils.modeling.model_paths import repo_root as _sdx_repo_root
from utils.prompt.rag import merge_facts_into_prompt
from utils.quality import test_time_pick as ttp

# ==================================================================================================
# inference_pipeline
# ==================================================================================================


@dataclass(slots=True)
class SuperiorInferenceConfig:
    """Knobs for a high-quality default sampling run."""

    num_candidates: int = 4
    pick_metric: str = "combo"
    use_composite_rank: bool = True
    cfg_scale: float = 7.0
    cfg_rescale: float = 0.3
    compile_inference: bool = False
    local_rag_jsonl: str | None = None
    rag_top_k: int = 8
    clip_guard_threshold: float = 0.0
    self_correct_clip: bool = False
    steps: int = 28
    fdg_low: float = 0.6
    fdg_high: float = 1.3
    feature_cache_delta: float = 0.0
    block_cache_threshold: float = 0.18
    taylor_cache: bool = True
    apg_parallel_eta: float = 0.0
    cfg_zero_star: bool = True
    qsilk_micrograin: float = 0.1


def build_superior_sample_argv(
    *,
    ckpt: str,
    prompt: str,
    out: str,
    config: SuperiorInferenceConfig | None = None,
) -> list[str]:
    """
    Return argv fragment for subprocess ``sample.py`` invocation (no executable name).

    Example::

        import subprocess, sys

        cfg = SuperiorInferenceConfig()
        argv = [
            sys.executable,
            "sample.py",
            *build_superior_sample_argv(ckpt="results/run/best.pt", prompt="a cat", out="out.png", config=cfg),
        ]
        subprocess.run(argv, check=True)
    """
    c = config or SuperiorInferenceConfig()
    args: list[str] = [
        "--ckpt",
        ckpt,
        "--prompt",
        prompt,
        "--out",
        out,
        "--num",
        str(max(1, c.num_candidates)),
        "--cfg-scale",
        str(c.cfg_scale),
        "--cfg-rescale",
        str(c.cfg_rescale),
        "--steps",
        str(c.steps),
    ]
    if c.num_candidates >= 2:
        metric = "superior_composite" if c.use_composite_rank else c.pick_metric
        args.extend(["--pick-best", metric])
    if c.compile_inference:
        args.append("--compile-inference")
    if c.local_rag_jsonl:
        args.extend(["--local-rag-jsonl", c.local_rag_jsonl, "--local-rag-top-k", str(c.rag_top_k)])
    if c.clip_guard_threshold > 0:
        args.extend(["--clip-guard-threshold", str(c.clip_guard_threshold)])
    if c.self_correct_clip:
        args.append("--superior-self-correct")
    if c.fdg_low != 1.0 or c.fdg_high != 1.0:
        args.extend(["--fdg-low", str(c.fdg_low), "--fdg-high", str(c.fdg_high)])
    if c.cfg_zero_star:
        args.append("--cfg-zero-star")
    if c.qsilk_micrograin > 0:
        args.extend(["--qsilk-micrograin", str(c.qsilk_micrograin)])
    if c.feature_cache_delta > 0:
        args.extend(["--feature-cache-delta", str(c.feature_cache_delta)])
    if c.block_cache_threshold > 0:
        args.extend(["--block-cache-thresh", str(c.block_cache_threshold)])
    if c.taylor_cache and c.block_cache_threshold > 0:
        args.append("--taylor-cache")
    if c.apg_parallel_eta >= 0:
        args.extend(["--apg-parallel-eta", str(c.apg_parallel_eta)])
    return args


# ==================================================================================================
# composite_ranker: The one candidate aggregator: weight presets over the ``test_time_pick`` terms.
#
# ``score_images`` ranks a candidate set by weighted Borda count (scale-free, see
# ``test_time_pick.rank_aggregate``). ``score_one`` gives an absolute [0, 1] score for a single
# image from unit-scaled terms, for callers that cannot compare a set: GRPO rollout rewards and
# agent verification. The ``reward`` term uses only HF reward models whose weights are on disk.
# ==================================================================================================

PRESETS: dict[str, dict[str, float]] = {
    # sample.py --pick-best superior_composite (the default profile's pick).
    "superior_composite": ttp.METRIC_WEIGHTS["superior_composite"],
    # Flow-GRPO terminal reward: alignment, natural look, art, learned preference.
    "online": {"clip": 0.25, "ai_look": 0.30, "art": 0.20, "reward": 0.15, "vit": 0.10},
    # Model-free (no CLIP / reward weights): loops that already spawn sample.py per candidate.
    "heuristic": {"ai_look": 0.45, "art": 0.35, "exposure": 0.20},
    # Cross-checkpoint ensemble pick; leans on the vit_quality head when one is given.
    "ensemble": {"clip": 0.20, "ai_look": 0.25, "art": 0.20, "reward": 0.15, "vit": 0.20},
}


class CompositeRanker:
    """Score RGB uint8 HWC images with a preset name or an explicit ``{term: weight}`` map."""

    def __init__(self, weights: str | Mapping[str, float] = "superior_composite") -> None:
        self.weights = dict(PRESETS[weights] if isinstance(weights, str) else weights)
        unknown = set(self.weights) - set(ttp.TERMS)
        if unknown:
            raise ValueError(f"unknown ranking terms: {sorted(unknown)}")

    def _context(self, prompt: str, device: str, vit_ckpt: str, plan: dict | None) -> ttp.PickContext:
        from utils.quality.critics import resolve_plan

        return ttp.PickContext(prompt=prompt, device=device, vit_ckpt_path=vit_ckpt, plan=resolve_plan(prompt, plan))

    def score_images(
        self,
        images: Sequence[np.ndarray],
        *,
        prompt: str = "",
        device: str = "cuda",
        vit_ckpt: str = "",
        plan: dict | None = None,
    ) -> list[float]:
        """Per-image weighted mean rank in [0, 1] (higher is better)."""
        if not images:
            return []
        ctx = self._context(prompt, device, vit_ckpt, plan)
        return ttp.rank_aggregate(ttp.term_scores(images, list(self.weights), ctx), self.weights)

    def pick_best_index(
        self,
        images: Sequence[np.ndarray],
        *,
        prompt: str = "",
        device: str = "cuda",
        vit_ckpt: str = "",
        plan: dict | None = None,
    ) -> tuple[int, list[float]]:
        scores = self.score_images(images, prompt=prompt, device=device, vit_ckpt=vit_ckpt, plan=plan)
        return int(np.argmax(scores)), scores

    def score_one(
        self,
        image: np.ndarray,
        *,
        prompt: str = "",
        device: str = "cuda",
        vit_ckpt: str = "",
        plan: dict | None = None,
    ) -> float:
        """Absolute [0, 1] score: weighted mean of the unit-scaled terms that have evidence."""
        ctx = self._context(prompt, device, vit_ckpt, plan)
        raw = ttp.term_scores([image], list(self.weights), ctx)
        vals = {k: ttp.unit_scale(k, v[0]) for k, v in raw.items() if v is not None}
        total = sum(self.weights[k] for k in vals)
        return float(sum(self.weights[k] * v for k, v in vals.items()) / total) if total > 0 else 0.5


def verify_metrics(
    image: np.ndarray, prompt: str, *, device: str = "cuda", vit_ckpt: str = "", expected_text: str = ""
) -> dict[str, float]:
    """Agent verification metrics: absolute composite, CLIP similarity, sharpness, aesthetic, OCR match."""
    clip = ttp.score_clip_similarity([image], prompt, device=device)
    metrics = {
        "composite": CompositeRanker().score_one(image, prompt=prompt, device=device, vit_ckpt=vit_ckpt),
        "clip": float(clip[0]) if clip else 0.0,
        "sharpness": float(ttp.score_edge_sharpness(image)),
        "aesthetic": float(ttp.score_aesthetic_proxy(image)),
    }
    if expected_text:
        metrics["ocr_match"] = float(ttp.score_ocr_match(image, expected_text))
    return metrics


# ==================================================================================================
# ensemble: **Ensemble inference** — generate from multiple checkpoints, pick global best candidate.
#
# Each checkpoint runs ``sample.py`` with ``--num 1`` (or shared num); all RGB outputs are
# ranked together by ``CompositeRanker`` (its ``ensemble`` preset).
# ==================================================================================================


@dataclass(slots=True)
class EnsembleConfig:
    checkpoints: list[str]
    weights: list[float] = field(default_factory=list)
    num_per_ckpt: int = 2
    pick_metric: str = "superior_composite"
    vit_ckpt: str = ""
    local_rag_jsonl: str = ""
    expand_prompt: bool = True
    self_correct: bool = False
    compile_inference: bool = False
    device: str = "cuda"


def _run_one_sample(
    repo_root: Path,
    ckpt: str,
    prompt: str,
    out_png: Path,
    *,
    device: str,
    extra_argv: Sequence[str],
) -> bool:
    cmd = [
        sys.executable,
        str(repo_root / "sample.py"),
        "--ckpt",
        ckpt,
        "--prompt",
        prompt,
        "--out",
        str(out_png),
        "--device",
        device,
    ]
    cmd.extend(extra_argv)
    rc = subprocess.run(cmd, cwd=str(repo_root), check=False).returncode
    return rc == 0 and out_png.is_file()


def generate_ensemble(
    prompt: str,
    cfg: EnsembleConfig,
    *,
    out: str = "ensemble_best.png",
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> tuple[Path | None, list[tuple[str, Path, float]]]:
    """
    Run all checkpoints, return ``(best_image_path, [(ckpt, path, score), ...])``.
    """
    root = Path(repo_root or _sdx_repo_root())
    if not cfg.checkpoints:
        raise ValueError("checkpoints must be non-empty")

    sup = SuperiorInferenceConfig(
        num_candidates=max(1, cfg.num_per_ckpt),
        pick_metric=cfg.pick_metric,
        use_composite_rank=cfg.pick_metric == "superior_composite",
        local_rag_jsonl=cfg.local_rag_jsonl or None,
        self_correct_clip=cfg.self_correct,
        compile_inference=cfg.compile_inference,
    )
    base_argv = build_superior_sample_argv(ckpt="PLACEHOLDER", prompt=prompt, out="PLACEHOLDER", config=sup)
    # Strip ckpt/out/prompt from base — we add per run
    skip = {"--ckpt", "--prompt", "--out"}
    extra: list[str] = []
    i = 0
    while i < len(base_argv):
        if base_argv[i] in skip:
            i += 2
            continue
        extra.append(base_argv[i])
        i += 1
    if cfg.expand_prompt:
        extra.append("--expand-prompt")

    scored: list[tuple[str, Path, np.ndarray]] = []

    with tempfile.TemporaryDirectory(prefix="sdx_ensemble_") as tmp:
        tmp_path = Path(tmp)
        for ci, ckpt in enumerate(cfg.checkpoints):
            for ni in range(max(1, cfg.num_per_ckpt)):
                out_png = tmp_path / f"ckpt{ci}_n{ni}.png"
                if dry_run:
                    print(f"would generate: {ckpt} -> {out_png}")
                    continue
                ok = _run_one_sample(root, ckpt, prompt, out_png, device=cfg.device, extra_argv=extra)
                if ok:
                    scored.append((ckpt, out_png, np.array(Image.open(out_png).convert("RGB"), dtype=np.uint8)))

        if dry_run or not scored:
            return None, []

        scores = CompositeRanker("ensemble").score_images(
            [arr for _, _, arr in scored], prompt=prompt, device=cfg.device, vit_ckpt=cfg.vit_ckpt
        )
        best_arr = scored[int(np.argmax(scores))][2]
        persist = Path(out)
        if not persist.is_absolute():
            persist = root / persist
        persist.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(best_arr).save(persist)
        return persist, [(ckpt, path, float(sc)) for (ckpt, path, _arr), sc in zip(scored, scores, strict=True)]


# ==================================================================================================
# retrieval: Local **RAG** without external APIs: TF-IDF retrieval over JSONL fact corpora.
#
# Pair with ``utils.prompt.rag.merge_facts_into_prompt`` to ground generation on your
# dataset captions, style notes, or agentic-search exports.
# ==================================================================================================

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]{1,}", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass(slots=True)
class TfidfFactIndex:
    """In-memory TF-IDF index over fact strings."""

    facts: list[str]
    doc_freq: dict[str, int] = field(default_factory=dict)
    doc_tfidf: list[dict[str, float]] = field(default_factory=list)
    n_docs: int = 0

    def __post_init__(self) -> None:
        self.n_docs = len(self.facts)
        if self.n_docs == 0:
            return
        if not self.doc_tfidf:
            self._build()

    def _build(self) -> None:
        self.doc_freq = {}
        tf_rows: list[Counter[str]] = []
        for fact in self.facts:
            toks = _tokenize(fact)
            c = Counter(toks)
            tf_rows.append(c)
            for tok in c:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1
        self.doc_tfidf = []
        for c in tf_rows:
            row: dict[str, float] = {}
            denom = float(sum(c.values()) or 1)
            for tok, cnt in c.items():
                idf = math.log((1.0 + self.n_docs) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
                row[tok] = (cnt / denom) * idf
            self.doc_tfidf.append(row)

    def query(self, text: str, *, top_k: int = 8) -> list[str]:
        """Return up to ``top_k`` facts ranked by cosine similarity to ``text``."""
        if not self.facts or top_k <= 0:
            return []
        q_toks = _tokenize(text)
        if not q_toks:
            return self.facts[:top_k]
        q_cnt = Counter(q_toks)
        q_denom = float(sum(q_cnt.values()) or 1)
        q_vec: dict[str, float] = {}
        for tok, cnt in q_cnt.items():
            idf = math.log((1.0 + self.n_docs) / (1.0 + self.doc_freq.get(tok, 0))) + 1.0
            q_vec[tok] = (cnt / q_denom) * idf
        scores: list[tuple[float, int]] = []
        for i, dvec in enumerate(self.doc_tfidf):
            dot = sum(q_vec.get(k, 0.0) * v for k, v in dvec.items())
            qn = math.sqrt(sum(v * v for v in q_vec.values())) + 1e-8
            dn = math.sqrt(sum(v * v for v in dvec.values())) + 1e-8
            scores.append((dot / (qn * dn), i))
        scores.sort(key=lambda x: -x[0])
        out: list[str] = []
        seen: set[str] = set()
        for _s, idx in scores[: max(top_k * 2, top_k)]:
            f = self.facts[idx]
            key = f.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= top_k:
                break
        return out


def build_tfidf_index_from_jsonl(
    path: str | Path,
    *,
    text_key: str = "text",
    max_entries: int = 50_000,
    extra_keys: Sequence[str] = ("caption", "summary", "description"),
) -> TfidfFactIndex:
    """Load facts from JSONL (and optional extra string fields per row)."""
    p = Path(path)
    facts: list[str] = []
    if not p.is_file():
        return TfidfFactIndex(facts=[])
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            parts: list[str] = []
            for k in (text_key, *extra_keys):
                v = obj.get(k)
                if v is not None and str(v).strip():
                    parts.append(str(v).strip())
            if parts:
                facts.append(" — ".join(parts))
            if len(facts) >= max_entries:
                break
    return TfidfFactIndex(facts=facts)


# ==================================================================================================
# auto_stack: **Superior prompt stack** — one call to enrich prompts before T5 encode.
#
# Layers: local TF-IDF RAG → domain tips → optional quality negatives.
# ==================================================================================================


@dataclass(slots=True)
class SuperiorPromptStack:
    """Configuration for automatic prompt enrichment."""

    rag_jsonl: str | None = None
    rag_top_k: int = 8
    rag_max_chars: int = 4000
    append_domain_tips: bool = True
    _index: TfidfFactIndex | None = field(default=None, repr=False)

    def _get_index(self) -> TfidfFactIndex | None:
        if not self.rag_jsonl:
            return None
        if self._index is None:
            self._index = build_tfidf_index_from_jsonl(self.rag_jsonl)
        return self._index

    def enrich(self, user_prompt: str) -> str:
        """Return prompt with retrieved facts and optional domain hints."""
        prompt = (user_prompt or "").strip()
        idx = self._get_index()
        if idx is not None and idx.n_docs > 0:
            facts = idx.query(prompt, top_k=self.rag_top_k)
            if facts:
                prompt = merge_facts_into_prompt(prompt, facts, max_chars=self.rag_max_chars)
        if self.append_domain_tips and "plastic skin" not in prompt.lower():
            prompt = prompt + ", natural skin texture, coherent anatomy, sharp focus"
        return prompt.strip()


# ==================================================================================================
# prompt_expand: Optional **prompt expansion** before generation (LLM or heuristic fallback).
# ==================================================================================================


@dataclass(slots=True)
class ExpandConfig:
    llm_model_path: str = ""
    device: str = "cuda"
    max_new_tokens: int = 128
    use_heuristic_fallback: bool = True


# Prompts this long are already specified; generic garnish only dilutes them.
_SKIP_EXPAND_WORDS = 32


def expand_prompt_heuristic(prompt: str) -> str:
    """Cheap expansion without LLM: add only the lighting/detail/composition cues a short prompt lacks."""
    p = (prompt or "").strip()
    if not p or len(p.split()) >= _SKIP_EXPAND_WORDS:
        return p
    low = p.lower()
    extras = []
    if "lighting" not in low:
        extras.append("natural lighting")
    if "detail" not in low:
        extras.append("highly detailed")
    if "composition" not in low:
        extras.append("balanced composition")
    return ", ".join([p, *extras])


def expand_prompt(
    prompt: str,
    cfg: ExpandConfig | None = None,
) -> str:
    """Expand prompt via LLM if configured, else heuristic."""
    c = cfg or ExpandConfig()
    p = (prompt or "").strip()
    if not p:
        return p
    if c.llm_model_path:
        try:
            from utils.agentic.agent import expand_prompt_qwen, load_qwen_causal_lm

            tok, model = load_qwen_causal_lm(c.llm_model_path, device=c.device)
            out = expand_prompt_qwen(tok, model, p, device=c.device, max_new_tokens=c.max_new_tokens)
            if out.strip():
                return out.strip()
        except Exception:
            pass
    if c.use_heuristic_fallback:
        return expand_prompt_heuristic(p)
    return p


# ==================================================================================================
# self_correct: **Self-correction** policy: CLIP alignment gates + optional rewind metadata for multi-pass sampling.
#
# Use with ``sample.py`` refinement flags or a custom loop that decodes latents between steps.
# ==================================================================================================


@dataclass(slots=True)
class RewindState:
    latent: torch.Tensor
    step_index: int
    timestep_value: int


def score_latent_prompt_alignment(
    latent: torch.Tensor,
    prompt: str,
    *,
    device: torch.device | None = None,
    clip_model_id: str = "",
    vae: Any = None,
    latent_scale: float = 1.0,
    ae_type: str = "kl",
    rae_bridge: Any = None,
) -> float:
    """CLIP cosine of the decoded latent vs ``prompt`` mapped to [0, 1]; 0.5 when CLIP or the VAE is missing."""
    if not str(clip_model_id).strip() or vae is None:
        return 0.5
    try:
        from utils.generation.guidance import latent_x0_clip_cosine

        c = latent_x0_clip_cosine(
            latent,
            prompt=prompt,
            model_id=str(clip_model_id),
            device=device if device is not None else latent.device,
            vae=vae,
            latent_scale=float(latent_scale),
            ae_type=str(ae_type),
            rae_bridge=rae_bridge,
            batch_index=0,
        )
        return float(max(0.0, min(1.0, (float(c) + 1.0) * 0.5)))
    except Exception:
        return 0.5


@dataclass(slots=True)
class SelfCorrectConfig:
    """Thresholds for alignment-based correction."""

    clip_model_id: str = "openai/clip-vit-base-patch32"
    align_threshold: float = 0.32
    """Mapped CLIP score in [0,1] below this triggers correction."""
    max_refinements: int = 1
    refine_steps: int = 8
    refine_t_frac: float = 0.12
    track_rewinds: bool = True


@dataclass(slots=True)
class SelfCorrectPolicy:
    """Stateful policy across denoise steps or full samples."""

    config: SelfCorrectConfig
    rewind_stack: list[RewindState] = field(default_factory=list)
    refinement_count: int = 0

    def score(
        self,
        latent: torch.Tensor,
        prompt: str,
        *,
        vae: Any,
        latent_scale: float,
        ae_type: str = "kl",
        rae_bridge: Any = None,
        device: torch.device | None = None,
    ) -> float:
        dev = device or latent.device
        return score_latent_prompt_alignment(
            latent,
            prompt,
            device=dev,
            clip_model_id=self.config.clip_model_id,
            vae=vae,
            latent_scale=latent_scale,
            ae_type=ae_type,
            rae_bridge=rae_bridge,
        )

    def needs_correction(self, score: float) -> bool:
        if self.refinement_count >= self.config.max_refinements:
            return False
        return score < self.config.align_threshold

    def push_rewind(self, latent: torch.Tensor, step_index: int, timestep_value: int) -> None:
        if not self.config.track_rewinds:
            return
        self.rewind_stack.append(
            RewindState(latent=latent.detach().clone(), step_index=step_index, timestep_value=timestep_value)
        )

    def pop_rewind(self) -> RewindState | None:
        if not self.rewind_stack:
            return None
        return self.rewind_stack.pop()

    def note_refinement_done(self) -> None:
        self.refinement_count += 1

    def refine_plan(self, num_timesteps: int) -> tuple[int, int]:
        """Return ``(t_start, num_steps)`` for a short re-denoise pass."""
        t_start = int(float(self.config.refine_t_frac) * max(1, num_timesteps - 1))
        t_start = min(max(1, t_start), num_timesteps - 1)
        return t_start, int(self.config.refine_steps)


# ==================================================================================================
# quality_gates: Pre/post generation **quality gates** (fast heuristics + optional CLIP).
#
# Use before accepting a candidate or after decode in multi-candidate flows.
# ==================================================================================================


@dataclass(slots=True)
class GateThresholds:
    """Floors on unit-scaled scores; ``min_sharpness`` 0.15 is a Laplacian variance of 60."""

    min_sharpness: float = 0.15
    min_exposure: float = 0.25
    min_clip: float = 0.0
    min_aesthetic: float = 0.0
    min_hf_reward: float = 0.0
    max_nsfw: float = 1.0
    max_watermark: float = 1.0
    clip_model_id: str = "openai/clip-vit-base-patch32"


@dataclass(slots=True)
class GateResult:
    passed: bool
    scores: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


class QualityGateRunner:
    """Run configurable gates on uint8 RGB images."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def evaluate(
        self,
        rgb_uint8: np.ndarray,
        *,
        prompt: str = "",
        device: str = "cpu",
    ) -> GateResult:
        th = self.thresholds
        scores: dict = {}
        failures: list[str] = []

        sharp = ttp.unit_scale("edge", ttp.score_edge_sharpness(rgb_uint8))
        scores["sharpness"] = sharp
        if sharp < th.min_sharpness:
            failures.append("sharpness")

        expo = ttp.score_exposure_balance(rgb_uint8)
        scores["exposure"] = expo
        if expo < th.min_exposure:
            failures.append("exposure")

        aes = ttp.score_aesthetic_proxy(rgb_uint8)
        scores["aesthetic"] = aes
        if th.min_aesthetic > 0 and aes < th.min_aesthetic:
            failures.append("aesthetic")

        if th.min_clip > 0 and prompt.strip():
            try:
                clip_scores = ttp.score_clip_similarity([rgb_uint8], prompt, device=device, model_id=th.clip_model_id)
                scores["clip"] = float(clip_scores[0]) if clip_scores else 0.0
                if scores["clip"] < th.min_clip:
                    failures.append("clip")
            except Exception:
                scores["clip"] = 0.0
                failures.append("clip")

        if th.min_hf_reward > 0:
            try:
                from utils.modeling.hf_loaders import HFRewardPanel

                hf_s = HFRewardPanel(device=device).score(rgb_uint8, prompt=prompt)
                scores["hf_reward"] = hf_s
                if hf_s < th.min_hf_reward:
                    failures.append("hf_reward")
            except Exception:
                pass

        if th.max_nsfw < 1.0:
            try:
                from utils.modeling.hf_loaders import score_nsfw_probability

                nsfw_p = score_nsfw_probability(rgb_uint8, device=device)
                if nsfw_p is not None:
                    scores["nsfw"] = float(nsfw_p)
                    if nsfw_p > th.max_nsfw:
                        failures.append("nsfw")
            except Exception:
                pass

        if th.max_watermark < 1.0:
            try:
                from utils.modeling.hf_loaders import score_watermark_probability

                wm_p = score_watermark_probability(rgb_uint8, device=device)
                if wm_p is not None:
                    scores["watermark"] = float(wm_p)
                    if wm_p > th.max_watermark:
                        failures.append("watermark")
            except Exception:
                pass

        return GateResult(passed=len(failures) == 0, scores=scores, failures=failures)


# ==================================================================================================
# auto_loop
# ==================================================================================================


@dataclass(slots=True)
class AutoImproveConfig:
    base_ckpt: str
    work_dir: str = "auto_improve_loop"
    preset: str = "sdxl"
    device: str = "cuda"
    suite_pack: str = "top_contender_proxy_v1"
    steps: int = 30
    num: int = 4
    pick_best: str = "superior_composite"
    iterations: int = 1
    dpo_steps: int = 500
    dpo_beta: float = 300.0
    vit_ckpt: str = ""
    local_rag_jsonl: str = ""
    model_soup: bool = False
    soup_weight_base: float = 0.35
    soup_weight_dpo: float = 0.65
    promote_best: bool = True
    promote_path: str = "auto_improve_loop/best_auto.pt"
    extra_args: list[str] = field(default_factory=list)


def build_auto_improve_argv(cfg: AutoImproveConfig) -> list[str]:
    argv = [
        "--base-ckpt",
        cfg.base_ckpt,
        "--work-dir",
        cfg.work_dir,
        "--preset",
        cfg.preset,
        "--device",
        cfg.device,
        "--suite-pack",
        cfg.suite_pack,
        "--steps",
        str(cfg.steps),
        "--num",
        str(cfg.num),
        "--pick-best",
        cfg.pick_best,
        "--iterations",
        str(cfg.iterations),
        "--dpo-steps",
        str(cfg.dpo_steps),
        "--dpo-beta",
        str(cfg.dpo_beta),
    ]
    if cfg.vit_ckpt:
        argv.extend(["--vit-ckpt", cfg.vit_ckpt])
    if cfg.local_rag_jsonl:
        argv.extend(["--local-rag-jsonl", cfg.local_rag_jsonl])
    if cfg.model_soup:
        argv.append("--model-soup")
        argv.extend(["--soup-weight-base", str(cfg.soup_weight_base)])
        argv.extend(["--soup-weight-dpo", str(cfg.soup_weight_dpo)])
    if cfg.promote_best:
        argv.append("--promote-best")
        argv.extend(["--promote-path", cfg.promote_path])
    argv.extend(cfg.extra_args)
    return argv


def run_tool(tool: str, argv: list[str], *, repo_root: str | Path | None = None, dry_run: bool = False) -> int:
    """Run ``python -m scripts.tools <tool> <argv>`` from the repo root; ``dry_run`` only prints it."""
    cmd = [sys.executable, "-m", "scripts.tools", tool, *argv]
    if dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.run(cmd, cwd=str(repo_root or _sdx_repo_root()), check=False).returncode


def run_auto_improve(cfg: AutoImproveConfig, *, repo_root: Path | None = None, dry_run: bool = False) -> int:
    return run_tool("auto_improve_loop", build_auto_improve_argv(cfg), repo_root=repo_root, dry_run=dry_run)


# ==================================================================================================
# dpo_pipeline: End-to-end **DPO alignment** helpers: mine preferences → train policy → export checkpoint.
#
# Wraps existing ``mine_preference_pairs``, ``train_diffusion_dpo``, and ``PreferenceImageDataset``.
# ==================================================================================================


@dataclass(slots=True)
class DPOStageConfig:
    ckpt: str
    preference_jsonl: str
    image_root: str = ""
    out: str = "dpo_policy.pt"
    steps: int = 500
    batch_size: int = 2
    lr: float = 1e-6
    dpo_beta: float = 300.0
    dpo_logit_clip: float = 40.0
    sync_ref_every: int = 0
    ref_ema_alpha: float = 0.01
    timestep_dpo_weight: str = "high_noise"
    timestep_dpo_power: float = 0.5
    safeguarded_dpo: float = 0.85
    device: str = "cuda"


@dataclass(slots=True)
class MinePairsConfig:
    benchmark_json: str
    out_jsonl: str
    min_margin: float = 0.08
    max_pairs_per_case: int = 2


def mine_pairs_from_benchmark(cfg: MinePairsConfig) -> list[dict[str, Any]]:
    """Load benchmark results JSON and mine win/lose pairs."""
    from scripts.tools.superior import mine_pairs

    path = Path(cfg.benchmark_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("results", payload.get("rows", []))
    pairs = mine_pairs(rows, min_margin=cfg.min_margin, max_pairs_per_case=cfg.max_pairs_per_case)
    out = Path(cfg.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return pairs


def build_dpo_train_argv(cfg: DPOStageConfig) -> list[str]:
    """Argv for ``python -m scripts.tools train_diffusion_dpo``."""
    return [
        "--ckpt",
        cfg.ckpt,
        "--preference-jsonl",
        cfg.preference_jsonl,
        "--image-root",
        cfg.image_root,
        "--out",
        cfg.out,
        "--steps",
        str(cfg.steps),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--dpo-beta",
        str(cfg.dpo_beta),
        "--dpo-logit-clip",
        str(cfg.dpo_logit_clip),
        "--sync-ref-every",
        str(cfg.sync_ref_every),
        "--ref-ema-alpha",
        str(cfg.ref_ema_alpha),
        "--timestep-dpo-weight",
        str(cfg.timestep_dpo_weight),
        "--timestep-dpo-power",
        str(cfg.timestep_dpo_power),
        "--safeguarded-dpo",
        str(cfg.safeguarded_dpo),
        "--device",
        cfg.device,
    ]


def run_dpo_training(
    cfg: DPOStageConfig,
    *,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> int:
    return run_tool("train_diffusion_dpo", build_dpo_train_argv(cfg), repo_root=repo_root, dry_run=dry_run)


def run_alignment_loop(
    *,
    benchmark_json: str,
    base_ckpt: str,
    pairs_jsonl: str,
    dpo_out: str,
    mine: MinePairsConfig | None = None,
    dpo: DPOStageConfig | None = None,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Mine pairs from benchmark → train DPO policy. Returns summary dict.
    """
    mine_cfg = mine or MinePairsConfig(benchmark_json=benchmark_json, out_jsonl=pairs_jsonl)
    mine_cfg.benchmark_json = benchmark_json
    mine_cfg.out_jsonl = pairs_jsonl
    pairs = mine_pairs_from_benchmark(mine_cfg)
    dpo_cfg = dpo or DPOStageConfig(ckpt=base_ckpt, preference_jsonl=pairs_jsonl, out=dpo_out)
    dpo_cfg.ckpt = base_ckpt
    dpo_cfg.preference_jsonl = pairs_jsonl
    dpo_cfg.out = dpo_out
    rc = run_dpo_training(dpo_cfg, repo_root=repo_root, dry_run=dry_run)
    return {"pairs_mined": len(pairs), "dpo_exit_code": rc, "dpo_out": dpo_out}


# ==================================================================================================
# flywheel: **Quality flywheel** orchestrator: curate manifest → auto-improve loop → promote best checkpoint.
#
# Does not run full ``train.py`` (too project-specific); focuses on data curation + alignment.
# ==================================================================================================


def curate_manifest(plan: FlywheelPlan, *, repo_root: Path, dry_run: bool = False) -> str | None:
    """Run ``superior_curate`` when ``manifest_in`` and ``manifest_out`` are set."""
    if plan.skip_curate or not plan.manifest_in or not plan.manifest_out:
        return plan.manifest_in
    d = plan.defaults
    argv = [
        plan.manifest_in,
        "--out",
        plan.manifest_out,
        "--dedup",
        d.dedup,
        "--min-caption-len",
        str(d.min_caption_len),
        "--max-caption-len",
        str(d.max_caption_len),
    ]
    rc = run_tool("superior_curate", argv, repo_root=repo_root, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"superior_curate failed with exit {rc}")
    return plan.manifest_out


def run_flywheel(
    plan: FlywheelPlan,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute flywheel steps. Returns summary dict written to ``work_dir/flywheel_summary.json``.
    """
    root = Path(repo_root or _sdx_repo_root())
    work = Path(plan.work_dir)
    if not dry_run:
        work.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"plan": _plan_to_dict(plan), "steps": []}

    if not plan.skip_curate and plan.manifest_in and plan.manifest_out:
        out = curate_manifest(plan, repo_root=root, dry_run=dry_run)
        summary["steps"].append({"curate": out})

    if not plan.skip_align:
        d = plan.defaults
        rag_jsonl = plan.local_rag_jsonl or ""
        cfg = AutoImproveConfig(
            base_ckpt=plan.base_ckpt,
            work_dir=str(work / "align"),
            preset=d.preset,
            suite_pack=d.benchmark_suite_pack,
            num=d.num_candidates,
            pick_best=d.pick_best,
            steps=d.steps,
            iterations=d.auto_loop_iterations,
            dpo_steps=d.dpo_steps,
            dpo_beta=d.dpo_beta,
            vit_ckpt=plan.vit_ckpt or "",
            local_rag_jsonl=rag_jsonl or "",
            model_soup=d.model_soup,
            soup_weight_base=d.soup_weight_base,
            soup_weight_dpo=d.soup_weight_dpo,
            promote_best=True,
            promote_path=plan.promote_path,
            extra_args=list(plan.extra_auto_loop_args),
        )
        rc = run_auto_improve(cfg, repo_root=root, dry_run=dry_run)
        summary["steps"].append({"align_exit_code": rc, "promote_path": plan.promote_path})
        if rc != 0 and not dry_run:
            summary["status"] = "align_failed"
            _write_summary(work, summary, dry_run)
            return summary

    summary["status"] = "ok"
    summary["best_ckpt"] = plan.promote_path
    _write_summary(work, summary, dry_run)
    return summary


def _plan_to_dict(plan: FlywheelPlan) -> dict[str, Any]:
    d = asdict(plan)
    d["defaults"] = asdict(plan.defaults)
    return d


def _write_summary(work: Path, summary: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(summary, indent=2))
        return
    (work / "flywheel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ==================================================================================================
# hard_negative: Mine **hard negative prompt tokens** from failed benchmark / hard-case exports.
#
# Appends domain-specific negatives to reduce recurring failure modes (blur, text, counting).
# ==================================================================================================

_TAG_NEGATIVES: dict[str, str] = {
    "low_composite": "low quality, blurry, artifacts",
    "text_rendering": "misspelled text, illegible lettering, garbled typography",
    "counting": "wrong number of objects, extra limbs, duplicated subjects",
    "exposure": "overexposed, underexposed, blown highlights, crushed shadows",
    "oversaturation": "oversaturated, neon colors, unrealistic skin tones",
    "blur": "blurry, out of focus, soft details, motion blur",
}


@dataclass(slots=True)
class HardNegativeBundle:
    tags: list[str]
    negative_suffix: str
    per_tag_counts: dict[str, int]


def tags_from_benchmark_row(row: dict[str, Any], *, threshold: float = 0.60) -> list[str]:
    """Infer failure tags from a benchmark ``results.json`` row (mirrors benchmark_suite)."""
    tags: list[str] = []
    comp = float(row.get("composite", 0.0) or 0.0)
    if comp < threshold:
        tags.append("low_composite")
    if float(row.get("ocr_match", 1.0) or 1.0) < 0.7 and row.get("expected_text"):
        tags.append("text_rendering")
    if float(row.get("count_match", 1.0) or 1.0) < 0.7 and int(row.get("expected_count", 0) or 0) > 0:
        tags.append("counting")
    if float(row.get("exposure_balance", 0.5) or 0.5) < 0.6:
        tags.append("exposure")
    if float(row.get("saturation_balance", 0.5) or 0.5) < 0.6:
        tags.append("oversaturation")
    edge = float(row.get("edge_sharpness", 0.0) or 0.0)
    if edge / 400.0 < 0.35:
        tags.append("blur")
    return tags


def mine_hard_negatives(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.60,
    max_phrases: int = 12,
) -> HardNegativeBundle:
    """Aggregate failure tags and build a negative prompt suffix."""
    counter: Counter[str] = Counter()
    for row in rows:
        for t in tags_from_benchmark_row(row, threshold=threshold):
            counter[t] += 1
        for t in row.get("failure_tags") or []:
            counter[str(t)] += 1
    ordered = [t for t, _ in counter.most_common(max_phrases)]
    suffix = merge_csv_unique(*(_TAG_NEGATIVES.get(tag, "") for tag in ordered))
    return HardNegativeBundle(tags=ordered, negative_suffix=suffix, per_tag_counts=dict(counter))


def load_hard_negatives_from_results(
    path: str | Path,
    *,
    threshold: float = 0.60,
) -> HardNegativeBundle:
    p = Path(path)
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = []
    low = [r for r in rows if isinstance(r, dict) and float(r.get("composite", 1.0) or 1.0) < threshold]
    return mine_hard_negatives(low or rows, threshold=threshold)


def benchmark_sample_args_for_negatives(bundle: HardNegativeBundle) -> list[str]:
    """Return ``benchmark_suite --sample-arg`` tokens for mined hard negatives."""
    if not bundle.negative_suffix.strip():
        return []
    return ["--negative-prompt", bundle.negative_suffix]


# ==================================================================================================
# vit_mining: **ViT-guided preference mining** — use ``vit_quality`` scores to rank candidates and mine DPO pairs.
#
# Falls back to benchmark ``composite`` when ViT checkpoint is unavailable.
# ==================================================================================================


@dataclass(slots=True)
class ViTMineConfig:
    vit_ckpt: str
    min_margin: float = 0.06
    max_pairs_per_case: int = 2
    vit_weight: float = 0.45
    composite_weight: float = 0.55
    device: str = "cuda"
    use_adherence: bool = True


def score_image_vit(
    image_path: str | Path,
    prompt: str,
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
) -> tuple[float, float]:
    """
    Return ``(quality_prob, adherence)`` in [0, 1]. On failure returns ``(0.5, 0.5)``.
    """
    try:
        import torch
        from vit_quality.model import cached_vit_scorer

        scorer = cached_vit_scorer(vit_ckpt, device)
        with Image.open(image_path) as pil:
            out = scorer(scorer.image_tensor(pil).unsqueeze(0), prompt)
        q = float(torch.sigmoid(out["quality_logit"]).item())
        a = float(out["adherence_score"].item())
        if use_adherence:
            return q, float(0.65 * q + 0.35 * a)
        return q, a
    except Exception:
        return 0.5, 0.5


def blended_reward(
    composite: float,
    vit_score: float,
    *,
    vit_weight: float = 0.45,
) -> float:
    cw = max(0.0, 1.0 - vit_weight)
    return float(cw * composite + vit_weight * vit_score)


def enrich_results_with_vit(
    rows: Sequence[dict[str, Any]],
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
    vit_weight: float = 0.45,
) -> list[dict[str, Any]]:
    """Add ``vit_quality``, ``vit_reward``, ``blended_score`` to each result row."""
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        path = str(row.get("output", "") or "")
        prompt = str(row.get("prompt", "") or "")
        comp = float(row.get("composite", 0.0) or 0.0)
        if path and Path(path).is_file():
            _q, vit_r = score_image_vit(path, prompt, vit_ckpt=vit_ckpt, device=device, use_adherence=use_adherence)
            row["vit_quality"] = _q
            row["vit_reward"] = vit_r
        else:
            row["vit_quality"] = 0.5
            row["vit_reward"] = 0.5
        row["blended_score"] = blended_reward(comp, float(row["vit_reward"]), vit_weight=vit_weight)
        out.append(row)
    return out


def mine_vit_preference_pairs(
    rows: Sequence[dict[str, Any]],
    cfg: ViTMineConfig,
) -> list[dict[str, Any]]:
    """Mine win/lose pairs ranked by ``blended_score`` (ViT + composite)."""
    enriched = enrich_results_with_vit(
        rows,
        vit_ckpt=cfg.vit_ckpt,
        device=cfg.device,
        use_adherence=cfg.use_adherence,
        vit_weight=cfg.vit_weight,
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in enriched:
        case = str(r.get("case", "") or "")
        prompt = str(r.get("prompt", "") or "")
        if not prompt:
            continue
        groups.setdefault((case, prompt), []).append(r)

    pairs: list[dict[str, Any]] = []
    for (case, prompt), items in groups.items():
        scored = sorted(items, key=lambda x: float(x.get("blended_score", 0.0)), reverse=True)
        if len(scored) < 2:
            continue
        hi, lo_best = scored[0], scored[-1]
        margin = float(hi.get("blended_score", 0)) - float(lo_best.get("blended_score", 0))
        if margin < cfg.min_margin:
            continue
        win_p = str(hi.get("output", "") or "")
        lose_p = str(lo_best.get("output", "") or "")
        if not win_p or not lose_p:
            continue
        pairs.append(
            {
                "win_image_path": win_p,
                "lose_image_path": lose_p,
                "caption": prompt,
                "case": case,
                "win_score": float(hi.get("blended_score", 0)),
                "lose_score": float(lo_best.get("blended_score", 0)),
                "margin": margin,
                "source": "vit_mining",
                "vit_weight": cfg.vit_weight,
            }
        )
        used = 1
        for lo in reversed(scored[1:-1]):
            if used >= cfg.max_pairs_per_case:
                break
            m = float(hi.get("blended_score", 0)) - float(lo.get("blended_score", 0))
            if m < cfg.min_margin:
                continue
            lp = str(lo.get("output", "") or "")
            if not lp:
                continue
            pairs.append(
                {
                    "win_image_path": win_p,
                    "lose_image_path": lp,
                    "caption": prompt,
                    "case": case,
                    "win_score": float(hi.get("blended_score", 0)),
                    "lose_score": float(lo.get("blended_score", 0)),
                    "margin": m,
                    "source": "vit_mining",
                    "vit_weight": cfg.vit_weight,
                }
            )
            used += 1
    return pairs


# ==================================================================================================
# model_soup: **Model soup**: average multiple DiT checkpoints for robust inference weights.
#
# Same architecture + compatible keys only; useful after multi-seed training or DPO + base blend.
# ==================================================================================================


def _load_state(path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(sd, dict):
        raise ValueError(f"No model/ema state in {path}")
    return {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu() for k, v in sd.items()}


def average_state_dicts(
    states: Sequence[dict[str, torch.Tensor]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Element-wise weighted average of compatible state dicts."""
    if not states:
        raise ValueError("states must be non-empty")
    keys = set(states[0].keys())
    for s in states[1:]:
        keys &= set(s.keys())
    if not keys:
        raise ValueError("No shared keys across checkpoints")
    w = list(weights) if weights is not None else [1.0 / len(states)] * len(states)
    if len(w) != len(states):
        raise ValueError("weights length must match states")
    ws = sum(w) or 1.0
    w = [x / ws for x in w]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for i, st in enumerate(states):
            t = st[k]
            if not t.is_floating_point():
                out[k] = t.clone()
                acc = None
                break
            part = t.float() * w[i]
            acc = part if acc is None else acc + part
        if acc is not None:
            out[k] = acc.to(dtype=states[0][k].dtype)
    return out


def soup_checkpoints(
    paths: Iterable[str | Path],
    *,
    weights: Sequence[float] | None = None,
    template_ckpt: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load and average multiple ``.pt`` checkpoints."""
    plist = [Path(p) for p in paths]
    states = [_load_state(p) for p in plist]
    return average_state_dicts(states, weights=weights)


def save_soup_checkpoint(
    averaged: dict[str, torch.Tensor],
    *,
    template_path: str | Path,
    out_path: str | Path,
    tag: str = "model_soup",
) -> Path:
    """Write soup weights using metadata from ``template_path``."""
    tpl = torch.load(str(template_path), map_location="cpu", weights_only=False)
    tpl["model"] = averaged
    tpl["ema"] = averaged
    meta = tpl.get("meta") if isinstance(tpl.get("meta"), dict) else {}
    meta["model_soup"] = tag
    tpl["meta"] = meta
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tpl, op)
    return op


# ==================================================================================================
# eval_report: Generate **Markdown eval reports** from ``benchmark_suite`` outputs.
# ==================================================================================================


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_scaffold_summary() -> list[str]:
    """Lines for eval report: local config scaffolds vs hub fallback."""
    try:
        from utils.modeling.model_paths import has_local_weights, model_dir, scaffold_registry

        local_cfg: list[str] = []
        local_w: list[str] = []
        for e in scaffold_registry():
            fp = model_dir() / e.name
            if not fp.is_dir() or not any(fp.iterdir()):
                continue
            if has_local_weights(fp):
                local_w.append(e.name)
            else:
                local_cfg.append(e.name)
        lines: list[str] = []
        if local_w:
            suffix = " …" if len(local_w) > 12 else ""
            lines.append(f"- Weights present ({len(local_w)}): {', '.join(local_w[:12])}{suffix}")
        if local_cfg:
            suffix = " …" if len(local_cfg) > 12 else ""
            lines.append(f"- Config-only scaffolds ({len(local_cfg)}): {', '.join(local_cfg[:12])}{suffix}")
        return lines
    except Exception:
        return []


def build_markdown_report(bench_dir: str | Path) -> str:
    """Build a human-readable report from ``results.json`` + ``leaderboard.json``."""
    d = Path(bench_dir)
    results = _load_json(d / "results.json") or []
    leaderboard = _load_json(d / "leaderboard.json") or []

    lines: list[str] = [
        "# SDX Benchmark Report",
        "",
        f"Directory: `{d}`",
        "",
        "## Leaderboard",
        "",
        "| Model | Mean composite | Std | Robust score | Cases |",
        "|-------|----------------|-----|--------------|-------|",
    ]
    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('model', '')} | {float(row.get('mean_composite', 0)):.3f} | "
            f"{float(row.get('std_composite', 0)):.3f} | {float(row.get('robust_score', 0)):.3f} | "
            f"{int(row.get('cases', 0))} |"
        )

    # Per-case worst scores
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        case = str(r.get("case", "") or "unknown")
        by_case.setdefault(case, []).append(r)

    lines.extend(["", "## Weakest cases (lowest composite)", ""])
    case_avgs: list[tuple[str, float]] = []
    for case, rows in by_case.items():
        vals = [float(x.get("composite", 0)) for x in rows]
        if vals:
            case_avgs.append((case, sum(vals) / len(vals)))
    case_avgs.sort(key=lambda x: x[1])
    for case, avg in case_avgs[:8]:
        lines.append(f"- **{case}**: avg composite {avg:.3f}")

    if results:
        lines.extend(["", "## Hard-negative hint", ""])
        try:
            bundle = load_hard_negatives_from_results(d / "results.json")
            if bundle.negative_suffix:
                lines.append(f"Suggested extra negatives: `{bundle.negative_suffix[:200]}`")
        except Exception:
            pass

    hf_lines = _hf_scaffold_summary()
    if hf_lines:
        lines.extend(["", "## HF pretrained status", ""])
        lines.extend(hf_lines)

    try:
        from utils.modeling.encoders import stack_download_hint, stack_status, stack_status_lines

        st = stack_status("penta")
        lines.extend(["", "## Text encoder stack (penta)", ""])
        lines.extend(stack_status_lines("penta"))
        if st.weights_count < len(st.slots):
            lines.append(f"- Download: `{stack_download_hint('penta')}`")
    except Exception:
        pass

    if results:
        modes = {
            str(r.get("text_encoder_mode", "")) for r in results if isinstance(r, dict) and r.get("text_encoder_mode")
        }
        modes.discard("")
        if modes:
            lines.extend(["", "## Checkpoint text encoder modes (from benchmark)", ""])
            for m in sorted(modes):
                try:
                    from utils.modeling.encoders import text_encoder_mode_label

                    lines.append(f"- **{m}**: {text_encoder_mode_label(m)}")
                except Exception:
                    lines.append(f"- **{m}**")

    try:
        from utils.modeling.model_paths import registry_summary

        s = registry_summary()
        lines.extend(
            [
                "",
                "## HF registry summary",
                "",
                f"- Total registry entries: {s['total_registry']}",
                f"- Local folders: {s['local_folders']} (weights: {s['with_weights']}, config-only: {s['config_only']})",
            ]
        )
        role_counts = s.get("role_counts", {})
        if isinstance(role_counts, dict) and role_counts:
            top = ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items()))
            lines.append(f"- By role: {top}")
    except Exception:
        pass

    try:
        from utils.modeling.hf_loaders import controlnet_has_weights, list_controlnet_types

        ctrl_ready = [t for t in list_controlnet_types() if controlnet_has_weights(t)]
        if ctrl_ready:
            lines.extend(["", "## ControlNet weights available", ""])
            lines.append(f"- {', '.join(ctrl_ready)}")
    except Exception:
        pass

    try:
        from utils.modeling.model_paths import FACE_RESTORE_MODELS, UPSCALE_MODELS, has_local_weights

        up = [n for n, path in UPSCALE_MODELS.items() if has_local_weights(path())]
        face = [n for n, path in FACE_RESTORE_MODELS.items() if has_local_weights(path())]
        if up or face:
            lines.extend(["", "## Upscale / face-restore weights", ""])
            if up:
                lines.append(f"- Upscale: {', '.join(up)}")
            if face:
                lines.append(f"- Face restore: {', '.join(face)}")
    except Exception:
        pass

    # Optional per-result critic / HF reward columns
    hf_keys = ("natural", "art", "hf_reward", "hpsv2", "pickscore", "clip_h14")
    if results and any(isinstance(r, dict) and any(k in r for k in hf_keys) for r in results):
        lines.extend(["", "## Critic and reward averages (when logged in results.json)", ""])
        for key in hf_keys:
            vals = [float(r[key]) for r in results if isinstance(r, dict) and r.get(key) is not None]
            if vals:
                lines.append(f"- **{key}**: mean {sum(vals) / len(vals):.3f} (n={len(vals)})")

    lines.append("")
    return "\n".join(lines)


def write_report(bench_dir: str | Path, out_path: str | Path) -> Path:
    md = build_markdown_report(bench_dir)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(md, encoding="utf-8")
    return op
