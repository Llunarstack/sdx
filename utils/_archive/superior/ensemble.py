"""
**Ensemble inference** — generate from multiple checkpoints, pick global best candidate.

Each checkpoint runs ``sample.py`` with ``--num 1`` (or shared num); all RGB outputs are
ranked together with ``CompositeRanker`` or ``UnifiedRewardScorer``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .composite_ranker import CompositeRanker
from .inference_pipeline import SuperiorInferenceConfig, build_superior_sample_argv
from .reward_scorer import UnifiedRewardScorer


@dataclass(slots=True)
class EnsembleConfig:
    checkpoints: List[str]
    weights: List[float] = field(default_factory=list)
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
    repo_root: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[Optional[Path], List[Tuple[str, Path, float]]]:
    """
    Run all checkpoints, return ``(best_image_path, [(ckpt, path, score), ...])``.
    """
    root = Path(repo_root or Path(__file__).resolve().parents[2])
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
    extra: List[str] = []
    i = 0
    while i < len(base_argv):
        if base_argv[i] in skip:
            i += 2
            continue
        extra.append(base_argv[i])
        i += 1
    if cfg.expand_prompt:
        extra.append("--expand-prompt")

    use_unified = bool(cfg.vit_ckpt) or cfg.pick_metric in ("vit", "combo_vit", "combo_vit_hq", "superior_composite")
    ranker = CompositeRanker()
    unified = UnifiedRewardScorer(vit_ckpt=cfg.vit_ckpt, device=cfg.device) if use_unified else None
    scored: List[Tuple[str, Path, np.ndarray, float]] = []

    with tempfile.TemporaryDirectory(prefix="sdx_ensemble_") as tmp:
        tmp_path = Path(tmp)
        for ci, ckpt in enumerate(cfg.checkpoints):
            for ni in range(max(1, cfg.num_per_ckpt)):
                out_png = tmp_path / f"ckpt{ci}_n{ni}.png"
                if dry_run:
                    print(f"would generate: {ckpt} -> {out_png}")
                    continue
                ok = _run_one_sample(root, ckpt, prompt, out_png, device=cfg.device, extra_argv=extra)
                if not ok:
                    continue
                arr = np.array(Image.open(out_png).convert("RGB"), dtype=np.uint8)
                if unified is not None:
                    sc = float(unified.score(arr, prompt=prompt))
                else:
                    scores = ranker.score_images([arr], prompt=prompt, device=cfg.device, vit_ckpt=cfg.vit_ckpt)
                    sc = float(scores[0])
                scored.append((ckpt, out_png, arr, sc))

        if dry_run or not scored:
            return None, []

        best_i = int(max(range(len(scored)), key=lambda j: scored[j][3]))
        best_ckpt, best_path, best_arr, best_score = scored[best_i]
        persist = Path(out)
        if not persist.is_absolute():
            persist = root / persist
        persist.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(best_arr).save(persist)
        listing = [(ckpt, path, sc) for ckpt, path, _arr, sc in scored]
        return persist, listing


__all__ = ["EnsembleConfig", "generate_ensemble"]
