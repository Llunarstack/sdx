"""
Quality-gap stack: the config-driven best-model policy plus the text-embedding adherence hook.

``apply_best_model_policy`` reads ``config.defaults.quality_policy`` and soft-fills args so sampling
knows what to do from the prompt alone (no flag cookbook).
"""

from __future__ import annotations

import re
from typing import Any

import torch

_COLOUR_WORD = re.compile(
    r"\b(?:red|orange|yellow|green|blue|purple|violet|pink|teal|gold|silver|black|white|brown|gr[ae]y)\b", re.I
)


def apply_best_model_policy(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-fill sampling args from compete-mode ``QualityPolicy``.

    Always-on unless ``args.no_quality_defaults`` or ``POLICY.enabled`` is False.
    Explicit CLI values always win (only unset fields are filled). The holy-grail preset it picks is
    expanded by the caller; it never picks ``--preset`` / ``--op-mode``, whose prompt packs are medium-blind.
    """
    from config.defaults.quality_policy import POLICY, recipe_for_prompt, soft_set

    meta: dict[str, Any] = {"profile": None, "applied": [], "aggression": POLICY.aggression}
    if not POLICY.enabled or bool(getattr(args, "no_quality_defaults", False)):
        meta["skipped"] = True
        return meta

    text = prompt if prompt is not None else str(getattr(args, "prompt", "") or "")
    style = str(getattr(args, "style", "") or "")
    recipe = recipe_for_prompt(text, style=style)
    meta["profile"] = recipe.name

    def _note(key: str, ok: bool) -> None:
        if ok:
            meta["applied"].append(key)

    if POLICY.enable_holy_grail and recipe.holy_grail:
        _note("holy_grail", soft_set(args, "holy_grail", True, unset=(False,)))
        _note(
            "holy_grail_preset",
            soft_set(args, "holy_grail_preset", recipe.holy_grail_preset, unset=(None, "")),
        )

    if recipe.cfg_scale is not None:
        _note("cfg_scale", soft_set(args, "cfg_scale", recipe.cfg_scale, unset=(0.0, 7.5, None)))
    _note("cfg_rescale", soft_set(args, "cfg_rescale", recipe.cfg_rescale, unset=(0.0, None)))
    if recipe.apg_parallel_eta is not None:
        _note("apg", soft_set(args, "apg_parallel_eta", recipe.apg_parallel_eta, unset=(-1.0, None)))
    if recipe.apg_momentum_beta is not None:
        _note(
            "apg_mom",
            soft_set(args, "apg_momentum_beta", recipe.apg_momentum_beta, unset=(0.0, None)),
        )
    if recipe.fdg_low is not None:
        _note("fdg_low", soft_set(args, "fdg_low", recipe.fdg_low, unset=(1.0, None)))
    if recipe.fdg_high is not None:
        _note("fdg_high", soft_set(args, "fdg_high", recipe.fdg_high, unset=(1.0, None)))
    if recipe.qsilk_micrograin is not None:
        _note(
            "qsilk",
            soft_set(args, "qsilk_micrograin", recipe.qsilk_micrograin, unset=(0.0, None)),
        )
    if recipe.steps is not None:
        _note("steps", soft_set(args, "steps", recipe.steps, unset=(0, 50, None)))

    # Photo-capture packs (skin pores, DSLR, "authentic photograph") only where a photograph is possible:
    # a named painting / anime / 3-D medium keeps its own look.
    from utils.prompt.art_direction import plan_art_direction

    ad = getattr(args, "_art_direction", None) or plan_art_direction(text)
    photo_ok = ad.medium_family not in ("2d", "traditional", "3d")
    if photo_ok:
        _note(
            "human_made",
            soft_set(args, "human_made", recipe.human_made, unset=("none", "off", "0", "", None)),
        )
    if recipe.less_ai and photo_ok:
        _note("less_ai", soft_set(args, "less_ai", True, unset=(False,)))
    if recipe.naturalize and photo_ok:
        _note("naturalize", soft_set(args, "naturalize", True, unset=(False,)))
    if recipe.naturalize_deep and photo_ok:
        _note("naturalize_deep", soft_set(args, "naturalize_deep", True, unset=(False,)))
    if photo_ok:
        _note(
            "anti_ai_pack",
            soft_set(args, "anti_ai_pack", recipe.anti_ai_pack, unset=("none", "", None)),
        )
    if recipe.human_media and photo_ok:
        _note(
            "human_media",
            soft_set(args, "human_media_mode", recipe.human_media, unset=("none", "", None)),
        )
    _note(
        "shortcomings",
        soft_set(
            args,
            "shortcomings_mitigation",
            recipe.shortcomings_mitigation,
            unset=("none", "", None),
        ),
    )

    anat = recipe.anatomy_guidance
    if anat == "auto":
        pl = text.lower()
        anat = (
            "strong"
            if any(k in pl for k in ("hand", "hands", "finger", "full body", "nude", "anatomy", "person", "people"))
            else "lite"
        )
    _note(
        "anatomy_guidance",
        soft_set(args, "anatomy_guidance", anat, unset=("none", "auto", "", None)),
    )
    if recipe.hand_mode and ad.has_people:
        _note(
            "hand_mode",
            soft_set(args, "hand_mode", recipe.hand_mode, unset=("none", "off", "", None)),
        )
    if recipe.pose_naturalness and ad.has_people:
        _note(
            "pose",
            soft_set(args, "pose_naturalness", recipe.pose_naturalness, unset=("none", "", None)),
        )
    if recipe.typography_mode:
        _note(
            "typography",
            soft_set(args, "typography_mode", recipe.typography_mode, unset=("none", "", None)),
        )

    if recipe.expand_prompt:
        _note("expand_prompt", soft_set(args, "expand_prompt", True, unset=(False,)))

    if recipe.photo_realism_prefer and ad.photographic:
        _note(
            "photo_pack",
            soft_set(
                args,
                "photo_realism_pack",
                recipe.photo_realism_prefer,
                unset=("none", "", None),
            ),
        )

    # Text-in-image + OCR auto
    needs_text = recipe.text_in_image or (
        POLICY.enable_ocr_auto and ('"' in text or "“" in text or "[text:" in text.lower() or "says" in text.lower())
    )
    if needs_text:
        _note("text_in_image", soft_set(args, "text_in_image", True, unset=(False,)))
        if recipe.ocr_fix or POLICY.enable_ocr_auto:
            _note("ocr_fix", soft_set(args, "ocr_fix", True, unset=(False,)))

    # Multi-candidate pick (compete mode)
    if POLICY.enable_pick_best and recipe.pick_best and POLICY.aggression == "max":
        if soft_set(args, "pick_best", recipe.pick_best, unset=("none", "", None)):
            _note("pick_best", True)
        if recipe.num is not None and soft_set(args, "num", int(recipe.num), unset=(1, None)):
            _note("num", True)

    # The subject planner pre-empts --frontier / --frontier-perfect / --frontier-creative, so it only
    # fills in when the user asked for none of them.
    frontier_asked = any(bool(getattr(args, f, False)) for f in ("frontier", "frontier_perfect", "frontier_creative"))
    if POLICY.enable_frontier_subject and recipe.frontier_subject and not frontier_asked:
        _note("frontier_subject", soft_set(args, "frontier_subject", True, unset=(False,)))
    if recipe.superior_self_correct:
        _note(
            "self_correct",
            soft_set(args, "superior_self_correct", True, unset=(False,)),
        )
    # The diversity pack is all faces, and the colour-separation pack only helps when two named colours
    # can bleed into each other; elsewhere they take the token budget from the scene.
    if recipe.diversity and ad.has_people:
        _note("diversity", soft_set(args, "diversity", True, unset=(False,)))

    if recipe.anti_slop and photo_ok:
        from frontier.subject import AntiSlopScanner

        # Only prompts that already ask for photorealism get the photo-tell packs.
        plan = AntiSlopScanner().plan(text)
        if plan.positive:
            cur = str(getattr(args, "prompt", "") or "")
            if plan.positive.lower() not in cur.lower():
                args.prompt = f"{cur}, {plan.positive}" if cur.strip() else plan.positive
                meta["applied"].append("anti_slop_pos")
        if plan.negative:
            base_neg = str(getattr(args, "negative_prompt", "") or "")
            if plan.negative.lower() not in base_neg.lower():
                args.negative_prompt = f"{base_neg}, {plan.negative}".strip(", ").strip()
                meta["applied"].append("anti_slop_neg")

    # Guidance interval + hard-token emphasis (compete mode)
    if POLICY.enable_cfg_interval and recipe.cfg_interval:
        _note("cfg_interval", soft_set(args, "cfg_interval", recipe.cfg_interval, unset=("", None)))
    if POLICY.enable_token_emphasis and recipe.token_emphasis:
        if not bool(getattr(args, "frontier_perfect", False)):
            from utils.generation.frontier_consume import apply_token_emphasis_to_args

            te = apply_token_emphasis_to_args(args)
            if "weights" in te:
                meta["applied"].append("token_emphasis")

    bleed_risk = len({m.group(0).lower() for m in _COLOUR_WORD.finditer(text)}) >= 2
    for flag in ("anti_bleed", "anti_artifacts", "strong_watermark"):
        want = getattr(recipe, flag, True) and (flag != "anti_bleed" or bleed_risk)
        if want and soft_set(args, flag, True, unset=(False,)):
            meta["applied"].append(flag)

    if POLICY.enable_box_attn:
        for flag in ("box_attn_layout", "per_region_cads"):
            if not hasattr(args, flag):
                setattr(args, flag, False)
            _note(flag, soft_set(args, flag, True, unset=(False,)))
    return meta


def _as_offset_pairs(raw: Any, seq_len: int) -> list[tuple[int, int]] | None:
    """Normalize tokenizer ``offset_mapping`` to ``[(start, end), ...]`` (unbatched)."""
    if raw is None:
        return None
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().tolist()
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not raw:
        return None
    first = raw[0]
    # Batched: [[(s, e), ...]] or [[[s, e], ...]]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple)):
        raw = first
    pairs: list[tuple[int, int]] = []
    for i, item in enumerate(raw):
        if i >= seq_len:
            break
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        a, b = item[0], item[1]
        if a is None or b is None:
            continue
        pairs.append((int(a), int(b)))
    return pairs or None


def _char_proportion_span(char_start: int, char_end: int, seq_len: int, prompt_len: int) -> slice:
    denom = max(int(prompt_len), 1)
    start = int(char_start * seq_len / denom)
    end = int(char_end * seq_len / denom)
    start = max(0, min(seq_len, start))
    end = max(0, min(seq_len, end))
    if end <= start:
        end = min(seq_len, start + 1)
    if start >= seq_len:
        start = max(0, seq_len - 1)
        end = seq_len
    return slice(start, end)


def _token_spans_for_words(prompt: str, seq_len: int, tokenizer: Any = None) -> list[slice]:
    """Map whitespace words to embedding-token slices (tokenizer offsets, else char proportion)."""
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []
    matches = list(re.finditer(r"\S+", prompt or ""))
    if not matches:
        return []
    prompt_len = max(len(prompt or ""), 1)

    def _fallback(cs: int, ce: int) -> slice:
        return _char_proportion_span(cs, ce, seq_len, prompt_len)

    offsets: list[tuple[int, int]] | None = None
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                prompt,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=seq_len,
            )
            raw = None
            if encoded is not None:
                if hasattr(encoded, "get"):
                    raw = encoded.get("offset_mapping")
                if raw is None:
                    raw = getattr(encoded, "offset_mapping", None)
            offsets = _as_offset_pairs(raw, seq_len)
        except Exception:
            offsets = None

    if not offsets:
        return [_fallback(m.start(), m.end()) for m in matches]

    spans: list[slice] = []
    for m in matches:
        ws, we = m.start(), m.end()
        hit = [i for i, (ts, te) in enumerate(offsets) if te > ts and ts < we and te > ws]
        if not hit:
            spans.append(_fallback(ws, we))
            continue
        start = max(0, min(hit))
        end = min(seq_len, max(hit) + 1)
        if end <= start:
            spans.append(_fallback(ws, we))
        else:
            spans.append(slice(start, end))
    return spans


def modulate_text_for_adherence(
    enc: torch.Tensor,
    prompt: str,
    *,
    negation_scale: float = 0.35,
    binding_boost: float = 0.08,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Training-free text-embedding modulation from ``PromptParser``."""
    if enc is None or enc.ndim != 3 or not (prompt or "").strip():
        return enc
    from models.prompt_adherence import PromptParser

    parsed = PromptParser().parse(prompt)
    out = enc.clone()
    _b, length, _d = out.shape
    spans = _token_spans_for_words(prompt, length, tokenizer)
    n_words = len(spans)
    if n_words == 0:
        return out

    damp = float(max(0.0, min(1.0, negation_scale)))
    if damp > 0.0:
        scale = 1.0 - damp
        for wi in getattr(parsed, "negation_indices", []) or []:
            for j in range(int(wi) + 1, min(int(wi) + 4, n_words)):
                sl = spans[j]
                if sl.start < sl.stop:
                    out[:, sl, :] = out[:, sl, :] * scale

    boost = float(max(0.0, min(0.5, binding_boost)))
    if boost > 0.0:
        gain = 1.0 + boost
        for triple in getattr(parsed, "triples", []) or []:
            for idx in getattr(triple, "token_indices", []) or []:
                ii = int(idx)
                if 0 <= ii < n_words:
                    sl = spans[ii]
                    if sl.start < sl.stop:
                        out[:, sl, :] = out[:, sl, :] * gain
        spatial = {
            "left",
            "right",
            "behind",
            "between",
            "beside",
            "holding",
            "under",
            "reflection",
            "exactly",
        }
        for i, match in enumerate(re.finditer(r"\S+", prompt or "")):
            if i >= n_words:
                break
            token = match.group(0).strip(".,;:").lower()
            if token in spatial:
                sl = spans[i]
                if sl.start < sl.stop:
                    out[:, sl, :] = out[:, sl, :] * (1.0 + 0.5 * boost)
    return out


__all__ = [
    "apply_best_model_policy",
    "modulate_text_for_adherence",
]
