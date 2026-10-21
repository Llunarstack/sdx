"""Build ``sample.py`` argument lists for **Superior** multi-candidate + ranking flows.

Also holds what used to be: composite_ranker, ensemble, retrieval, auto_stack, prompt_expand, self_correct, quality_gates.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from utils.modeling.model_paths import repo_root as _sdx_repo_root
from utils.prompt.rag_prompt import merge_facts_into_prompt
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
        from utils.superior.inference_pipeline import build_superior_sample_argv, SuperiorInferenceConfig

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
        from utils.quality.art_critic import resolve_plan

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
        return persist, [(ckpt, path, float(sc)) for (ckpt, path, _arr), sc in zip(scored, scores)]


# ==================================================================================================
# retrieval: Local **RAG** without external APIs: TF-IDF retrieval over JSONL fact corpora.
#
# Pair with ``utils.prompt.rag_prompt.merge_facts_into_prompt`` to ground generation on your
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
        from utils.generation.clip_alignment import latent_x0_clip_cosine

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
                from utils.modeling.hf_reward import HFRewardPanel

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


__all__ = [
    "SuperiorInferenceConfig",
    "build_superior_sample_argv",
    "PRESETS",
    "CompositeRanker",
    "verify_metrics",
    "EnsembleConfig",
    "generate_ensemble",
    "TfidfFactIndex",
    "build_tfidf_index_from_jsonl",
    "SuperiorPromptStack",
    "ExpandConfig",
    "expand_prompt",
    "expand_prompt_heuristic",
    "SelfCorrectConfig",
    "SelfCorrectPolicy",
    "GateResult",
    "GateThresholds",
    "QualityGateRunner",
]
