"""
Quality-gap stack: config-driven best-model policy + text/DiT hooks.

``apply_quality_defaults`` / ``apply_best_model_policy`` read
``config.defaults.quality_policy`` and soft-fill args so sampling knows what to
do from the prompt alone (no flag cookbook).
"""

from __future__ import annotations

import re
from typing import Any

import torch

_TEXT_IN_IMAGE = re.compile(
    r"""(?:text\s*(?:says|reading|that\s+says)|["“][^"”]{1,48}["”]|\[text:[^\]]+\])""",
    re.I,
)


def prompt_looks_photoreal(prompt: str) -> bool:
    from frontier.realism.anti_slop import AntiSlopScanner, RealismTier

    return AntiSlopScanner().detect_tier(prompt or "") != RealismTier.NONE


def prompt_needs_glyph(prompt: str) -> bool:
    return bool(_TEXT_IN_IMAGE.search(prompt or ""))


def apply_best_model_policy(args: Any, *, prompt: str | None = None) -> dict[str, Any]:
    """
    Soft-fill sampling args from compete-mode ``QualityPolicy``.

    Always-on unless ``args.no_quality_defaults`` or ``POLICY.enabled`` is False.
    Explicit CLI values always win (only unset fields are filled).
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
    args._quality_profile = recipe.name
    args._adherence_negation_scale = float(recipe.adherence_negation_scale)
    args._adherence_binding_boost = float(recipe.adherence_binding_boost)

    def _note(key: str, ok: bool) -> None:
        if ok:
            meta["applied"].append(key)

    if recipe.sampler_preset and soft_set(args, "preset", recipe.sampler_preset, unset=(None, "")):
        _note("preset", True)
    if recipe.op_mode and soft_set(args, "op_mode", recipe.op_mode, unset=(None, "")):
        _note("op_mode", True)

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
    if recipe.fdg_cfg_strength is not None:
        _note("fdg", soft_set(args, "fdg_cfg_strength", recipe.fdg_cfg_strength, unset=(0.0, None)))
    if recipe.zeresfdg_strength is not None:
        _note(
            "zeresfdg",
            soft_set(args, "zeresfdg_strength", recipe.zeresfdg_strength, unset=(0.0, None)),
        )
    if recipe.qsilk_micrograin is not None:
        _note(
            "qsilk",
            soft_set(args, "qsilk_micrograin", recipe.qsilk_micrograin, unset=(0.0, None)),
        )
    if recipe.steps is not None:
        _note("steps", soft_set(args, "steps", recipe.steps, unset=(0, 50, None)))

    _note(
        "human_made",
        soft_set(args, "human_made", recipe.human_made, unset=("none", "off", "0", "", None)),
    )
    if recipe.less_ai:
        _note("less_ai", soft_set(args, "less_ai", True, unset=(False,)))
    if recipe.naturalize:
        _note("naturalize", soft_set(args, "naturalize", True, unset=(False,)))
    if recipe.naturalize_deep:
        _note("naturalize_deep", soft_set(args, "naturalize_deep", True, unset=(False,)))
    _note(
        "anti_ai_pack",
        soft_set(args, "anti_ai_pack", recipe.anti_ai_pack, unset=("none", "", None)),
    )
    if recipe.human_media:
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
    if recipe.hand_mode:
        _note(
            "hand_mode",
            soft_set(args, "hand_mode", recipe.hand_mode, unset=("none", "off", "", None)),
        )
    if recipe.pose_naturalness:
        _note(
            "pose",
            soft_set(args, "pose_naturalness", recipe.pose_naturalness, unset=("none", "", None)),
        )
    if recipe.typography_mode:
        _note(
            "typography",
            soft_set(args, "typography_mode", recipe.typography_mode, unset=("none", "", None)),
        )

    _note(
        "naturalness_strength",
        soft_set(args, "naturalness_strength", recipe.naturalness_strength, unset=(-1.0, None)),
    )
    _note(
        "anatomy_attention_strength",
        soft_set(
            args,
            "anatomy_attention_strength",
            recipe.anatomy_attention_strength,
            unset=(-1.0, None),
        ),
    )
    _note(
        "glyph_residual_strength",
        soft_set(args, "glyph_residual_strength", recipe.glyph_residual_strength, unset=(-1.0, None)),
    )
    if recipe.expand_prompt:
        _note("expand_prompt", soft_set(args, "expand_prompt", True, unset=(False,)))

    if recipe.photo_realism_prefer:
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

    if POLICY.enable_frontier_subject and recipe.frontier_subject:
        _note("frontier_subject", soft_set(args, "frontier_subject", True, unset=(False,)))
    if recipe.superior_self_correct:
        _note(
            "self_correct",
            soft_set(args, "superior_self_correct", True, unset=(False,)),
        )
    if recipe.diversity:
        _note("diversity", soft_set(args, "diversity", True, unset=(False,)))

    if recipe.anti_slop:
        from frontier.realism.anti_slop import AntiSlopScanner, RealismTier

        plan = AntiSlopScanner().plan(text if text.strip() else "photoreal dslr photo")
        if plan.tier == RealismTier.NONE:
            plan = AntiSlopScanner().plan("photoreal dslr portrait photo")
        skip_prompt = bool(getattr(args, "frontier_perfect", False) or getattr(args, "frontier_subject", False))
        # Still inject anti-slop even with frontier_subject — subject plan doesn't replace skin tells
        if plan.positive:
            cur = str(getattr(args, "prompt", "") or "")
            if plan.positive.lower() not in cur.lower():
                args.prompt = f"{cur}, {plan.positive}" if cur.strip() else plan.positive
                meta["applied"].append("anti_slop_pos")
            if plan.microdetail_hint and plan.microdetail_hint.lower() not in str(args.prompt).lower():
                args.prompt = f"{args.prompt}, {plan.microdetail_hint}"
        if plan.negative:
            base_neg = str(getattr(args, "negative_prompt", "") or "")
            if plan.negative.lower() not in base_neg.lower():
                args.negative_prompt = f"{base_neg}, {plan.negative}".strip(", ").strip()
                meta["applied"].append("anti_slop_neg")
        _ = skip_prompt  # kept for future frontier merge

    # Guidance interval + hard-token emphasis (compete mode)
    if POLICY.enable_cfg_skip_early and recipe.cfg_skip_early_frac is not None:
        _note(
            "cfg_skip_early",
            soft_set(args, "cfg_skip_early_frac", float(recipe.cfg_skip_early_frac), unset=(0.0, None)),
        )
    if recipe.cfg_skip_late_frac is not None:
        _note(
            "cfg_skip_late",
            soft_set(args, "cfg_skip_late_frac", float(recipe.cfg_skip_late_frac), unset=(0.0, None)),
        )
    if POLICY.enable_token_emphasis and recipe.token_emphasis:
        if not bool(getattr(args, "frontier_perfect", False)):
            from utils.generation.frontier_consume import apply_token_emphasis_to_args

            te = apply_token_emphasis_to_args(args)
            if "weights" in te:
                meta["applied"].append("token_emphasis")

    for flag in ("anti_bleed", "anti_artifacts", "strong_watermark"):
        want = getattr(recipe, flag, True)
        if want and soft_set(args, flag, True, unset=(False,)):
            meta["applied"].append(flag)

    if POLICY.enable_auto_layout and not bool(getattr(args, "no_auto_layout", False)):
        _note("auto_layout", soft_set(args, "auto_layout", True, unset=(False,)))
    if POLICY.enable_prompt_ground and not bool(getattr(args, "no_prompt_ground", False)):
        _note("prompt_ground", soft_set(args, "prompt_ground", True, unset=(False,)))
    if POLICY.enable_glyph_canvas and not bool(getattr(args, "no_glyph_canvas", False)):
        _note("glyph_canvas", soft_set(args, "glyph_canvas", True, unset=(False,)))
    if POLICY.enable_contact_shadow and not bool(getattr(args, "no_contact_shadow", False)):
        _note("contact_shadow", soft_set(args, "contact_shadow_auto", True, unset=(False,)))
    if getattr(POLICY, "enable_design_brief", True) and not bool(getattr(args, "no_design_brief", False)):
        extra = str(getattr(args, "palette", "") or "").strip()
        prompt_has_hex = "#" in str(text or "")
        if extra or prompt_has_hex or '"' in str(text or "") or "[text:" in str(text or "").lower():
            _note("design_brief", True)
    if getattr(POLICY, "enable_community_recipes", True) and not bool(getattr(args, "no_community_recipes", False)):
        _note("community_recipes", soft_set(args, "community_recipes", "auto", unset=(None, "")))

    no_auto = bool(getattr(args, "no_auto_layout", False))
    auto_on = bool(getattr(args, "auto_layout", False)) or (POLICY.enable_auto_layout and not no_auto)
    if (POLICY.enable_box_attn or auto_on) and not no_auto and not bool(getattr(args, "no_box_attn_layout", False)):
        if not hasattr(args, "box_attn_layout"):
            args.box_attn_layout = False
        if not hasattr(args, "per_region_cads"):
            args.per_region_cads = False
        _note("box_attn_layout", soft_set(args, "box_attn_layout", True, unset=(False,)))
        _note("per_region_cads", soft_set(args, "per_region_cads", True, unset=(False,)))

    if POLICY.enable_prompt_reinject:
        check = str(getattr(args, "prompt", "") or text or "")
        n_words = len(check.split())
        n_chars = len(check)
        if not hasattr(args, "prompt_early_scale"):
            args.prompt_early_scale = -1
        if not hasattr(args, "prompt_reinject_every_n"):
            args.prompt_reinject_every_n = 0
        if not hasattr(args, "prompt_reinject_alpha"):
            args.prompt_reinject_alpha = 0.0
        if n_words > 48 or n_chars > 300:
            _note("prompt_early_scale", soft_set(args, "prompt_early_scale", 1.18))
            _note("prompt_reinject_every_n", soft_set(args, "prompt_reinject_every_n", 3))
            _note("prompt_reinject_alpha", soft_set(args, "prompt_reinject_alpha", 0.10))
        elif n_words > 28 or n_chars > 200:
            _note("prompt_early_scale", soft_set(args, "prompt_early_scale", 1.12))
            _note("prompt_reinject_every_n", soft_set(args, "prompt_reinject_every_n", 4))
            _note("prompt_reinject_alpha", soft_set(args, "prompt_reinject_alpha", 0.08))

    return meta


def apply_quality_defaults(args: Any) -> dict[str, Any]:
    """Backward-compatible alias for ``apply_best_model_policy``."""
    return apply_best_model_policy(args)


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


# Process-wide glyph projector cache: (glyph_dim, text_dim) -> GlyphToCondProjector
_GLYPH_PROJECTORS: dict[tuple[int, int], Any] = {}


def get_persistent_glyph_projector(
    glyph_dim: int,
    text_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint: str | None = None,
):
    """
    Cached ``GlyphToCondProjector``. Unlike a fresh sin/cos map each call, weights
    persist for the process (and can be loaded from ``checkpoint``).
    """
    from utils.generation.inference_research_hooks import GlyphToCondProjector

    key = (int(glyph_dim), int(text_dim))
    proj = _GLYPH_PROJECTORS.get(key)
    if proj is None:
        proj = GlyphToCondProjector(glyph_dim, text_dim)
        # Non-zero init so inference is useful before training (small orthogonal-ish).
        with torch.no_grad():
            w = proj.proj.weight
            nn_init = torch.nn.init
            nn_init.normal_(w, std=0.02)
            if proj.proj.bias is not None:
                nn_init.zeros_(proj.proj.bias)
        if checkpoint:
            try:
                sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "proj.weight" in sd:
                    proj.load_state_dict(sd, strict=False)
                elif isinstance(sd, dict) and any(k.startswith("proj.") for k in sd):
                    proj.load_state_dict(sd, strict=False)
            except Exception as e:
                import warnings

                warnings.warn(f"glyph projector checkpoint load failed ({checkpoint!r}): {e}", stacklevel=2)
        _GLYPH_PROJECTORS[key] = proj
    return proj.to(device=device, dtype=dtype)


def inject_glyph_residual(
    enc: torch.Tensor,
    prompt: str,
    *,
    strength: float = 0.06,
    texts: list[str] | None = None,
    glyph_checkpoint: str | None = None,
) -> torch.Tensor:
    """Add byte-hash glyph residual via a persistent projector into T5 states."""
    if enc is None or enc.ndim != 3:
        return enc
    if strength <= 0.0:
        return enc
    if texts is None:
        if not prompt_needs_glyph(prompt):
            return enc
        texts = [prompt]
    from utils.superior.glyph_encoder import ByteHashGlyphEncoder

    b, length, dim = enc.shape
    device = enc.device
    glyph_dim = min(64, dim)
    encoder = ByteHashGlyphEncoder(embed_dim=glyph_dim, max_bytes=min(256, length))
    glyphs = encoder.encode_utf8(list(texts)[:b], device=device)
    if glyphs.shape[0] == 1 and b > 1:
        glyphs = glyphs.expand(b, -1, -1)
    lg = glyphs.shape[1]
    projector = get_persistent_glyph_projector(
        glyph_dim, dim, device=device, dtype=enc.dtype, checkpoint=glyph_checkpoint
    )
    mapped = projector(glyphs.to(dtype=enc.dtype))
    if lg < length:
        pad = mapped.new_zeros(b, length - lg, dim)
        mapped = torch.cat([mapped, pad], dim=1)
    elif lg > length:
        mapped = mapped[:, :length, :]
    return enc + float(strength) * mapped


def enrich_text_conditioning(
    cond_emb: torch.Tensor,
    prompt: str,
    *,
    enable_adherence: bool = True,
    enable_glyph: bool = True,
    glyph_strength: float = 0.06,
    expected_texts: list[str] | None = None,
    negation_scale: float | None = None,
    binding_boost: float | None = None,
    tokenizer: Any = None,
) -> torch.Tensor:
    """Apply adherence modulation then optional glyph residual."""
    from config.defaults.quality_policy import POLICY

    out = cond_emb
    if enable_adherence and POLICY.enable_adherence_modulate:
        neg = float(negation_scale) if negation_scale is not None else float(POLICY.adherence_negation_scale)
        bind = float(binding_boost) if binding_boost is not None else float(POLICY.adherence_binding_boost)
        out = modulate_text_for_adherence(out, prompt, negation_scale=neg, binding_boost=bind, tokenizer=tokenizer)
    if enable_glyph and POLICY.enable_glyph:
        texts = None
        if expected_texts:
            texts = [", ".join(str(t) for t in expected_texts)]
        if expected_texts or prompt_needs_glyph(prompt) or texts is not None:
            gs = max(float(glyph_strength), 0.0)
            out = inject_glyph_residual(out, prompt, strength=gs, texts=texts)
    return out


def dit_quality_kwargs(args: Any, prompt: str) -> dict[str, Any]:
    """Kwargs merged into model_kwargs_cond for mid-network quality hooks."""
    from config.defaults.quality_policy import recipe_for_prompt

    kw: dict[str, Any] = {}
    recipe = recipe_for_prompt(prompt, style=str(getattr(args, "style", "") or ""))

    nat = float(getattr(args, "naturalness_strength", -1.0))
    if nat < 0.0:
        nat = float(recipe.naturalness_strength) if recipe.naturalness_strength >= 0 else 0.0
        if nat <= 0.0:
            hm = str(getattr(args, "human_made", "none") or "none").lower()
            if bool(getattr(args, "less_ai", False)) or hm not in ("none", "off", "0", ""):
                nat = 0.34
    if nat > 0.0:
        from models.anti_ai_naturalness import detect_medium

        kw["naturalness_strength"] = float(nat)
        kw["naturalness_medium"] = detect_medium(prompt)
        kw["naturalness_prompt"] = prompt

    anat_flag = str(getattr(args, "anatomy_guidance", "none") or "none").lower()
    anat_s = float(getattr(args, "anatomy_attention_strength", -1.0))
    if anat_s < 0.0:
        if anat_flag in ("auto",):
            pl = prompt.lower()
            anat_s = (
                0.35
                if any(k in pl for k in ("hand", "hands", "finger", "anatomy"))
                else 0.22
                if any(k in pl for k in ("person", "man", "woman", "girl", "boy", "portrait", "people"))
                else 0.0
            )
        elif anat_flag == "strong":
            anat_s = 0.35
        elif anat_flag == "lite":
            pl = prompt.lower()
            anat_s = (
                0.22
                if any(k in pl for k in ("person", "man", "woman", "girl", "boy", "hand", "portrait", "people"))
                else 0.0
            )
        else:
            anat_s = 0.0
    if anat_s > 0.0:
        kw["anatomy_attention_strength"] = float(anat_s)

    early = float(getattr(args, "prompt_early_scale", -1))
    if early > 0:
        kw["prompt_early_scale"] = early
        kw["prompt_timestep_schedule_enabled"] = True
    every_n = int(getattr(args, "prompt_reinject_every_n", 0) or 0)
    if every_n > 0:
        kw["prompt_reinject_every_n"] = every_n
        alpha = float(getattr(args, "prompt_reinject_alpha", 0.0) or 0.0)
        if alpha > 0:
            kw["prompt_reinject_alpha"] = alpha
    return kw


__all__ = [
    "prompt_looks_photoreal",
    "prompt_needs_glyph",
    "apply_best_model_policy",
    "apply_quality_defaults",
    "modulate_text_for_adherence",
    "inject_glyph_residual",
    "get_persistent_glyph_projector",
    "enrich_text_conditioning",
    "dit_quality_kwargs",
]
