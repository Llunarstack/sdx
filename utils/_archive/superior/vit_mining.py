"""
**ViT-guided preference mining** — use ``vit_quality`` scores to rank candidates and mine DPO pairs.

Falls back to benchmark ``composite`` when ViT checkpoint is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass(slots=True)
class ViTMineConfig:
    vit_ckpt: str
    min_margin: float = 0.06
    max_pairs_per_case: int = 2
    vit_weight: float = 0.45
    composite_weight: float = 0.55
    device: str = "cuda"
    use_adherence: bool = True


_vit_bundle: Optional[Tuple[Tuple[Any, Any, Any], Any, str]] = None


def _get_vit(vit_ckpt: str, device: str) -> Tuple[Tuple[Any, Any, Any], Any]:
    global _vit_bundle
    if _vit_bundle is not None and _vit_bundle[2] == vit_ckpt:
        return _vit_bundle[0], _vit_bundle[1]
    import torch
    from torchvision import transforms
    from vit_quality.checkpoint_utils import load_vit_quality_checkpoint
    from vit_quality.dataset import text_feature_vector

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model, cfg = load_vit_quality_checkpoint(vit_ckpt)
    model = model.to(dev).eval()
    img_sz = int(cfg.get("image_size", 224) or 224)
    tfm = transforms.Compose(
        [
            transforms.Resize((img_sz, img_sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    bundle = (model, tfm, text_feature_vector)
    _vit_bundle = (bundle, dev, vit_ckpt)
    return bundle, dev


def score_image_vit(
    image_path: str | Path,
    prompt: str,
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
) -> Tuple[float, float]:
    """
    Return ``(quality_prob, adherence)`` in [0, 1]. On failure returns ``(0.5, 0.5)``.
    """
    try:
        import torch

        (model, tfm, text_feat_fn), dev = _get_vit(vit_ckpt, device)
        pil = Image.open(image_path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(dev)
        txt = text_feat_fn(prompt).unsqueeze(0).to(dev)
        with torch.inference_mode():
            out = model(x, txt)
            q = float(torch.sigmoid(out["quality_logit"]).item())
            a = float(out["adherence_score"].item()) if "adherence_score" in out else 0.5
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
    rows: Sequence[Dict[str, Any]],
    *,
    vit_ckpt: str,
    device: str = "cuda",
    use_adherence: bool = True,
    vit_weight: float = 0.45,
) -> List[Dict[str, Any]]:
    """Add ``vit_quality``, ``vit_reward``, ``blended_score`` to each result row."""
    out: List[Dict[str, Any]] = []
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
    rows: Sequence[Dict[str, Any]],
    cfg: ViTMineConfig,
) -> List[Dict[str, Any]]:
    """Mine win/lose pairs ranked by ``blended_score`` (ViT + composite)."""
    enriched = enrich_results_with_vit(
        rows,
        vit_ckpt=cfg.vit_ckpt,
        device=cfg.device,
        use_adherence=cfg.use_adherence,
        vit_weight=cfg.vit_weight,
    )
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in enriched:
        case = str(r.get("case", "") or "")
        prompt = str(r.get("prompt", "") or "")
        if not prompt:
            continue
        groups.setdefault((case, prompt), []).append(r)

    pairs: List[Dict[str, Any]] = []
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


def mine_vit_from_results_json(
    results_path: str | Path,
    out_jsonl: str | Path,
    cfg: ViTMineConfig,
) -> int:
    """Load ``results.json``, mine ViT-weighted pairs, write JSONL."""
    path = Path(results_path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        rows = []
    pairs = mine_vit_preference_pairs(rows, cfg)
    op = Path(out_jsonl)
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


__all__ = [
    "ViTMineConfig",
    "blended_reward",
    "enrich_results_with_vit",
    "mine_vit_from_results_json",
    "mine_vit_preference_pairs",
    "score_image_vit",
]
