"""Prompt phases of ``sample.py``: the user's prompt, prompt additions, and the final encoder text."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.generation.sample_helpers import (
    TEXT_MAX_TOKENS,
    _apply_gender_swap,
    _build_size_tokens,
    _infer_expected_texts_from_prompt,
    _load_character_sheet,
    _maybe_append_text_says,
    _parse_expected_texts,
    _parse_scale_csv,
)
from utils.prompt.prompt_emphasis import parse_prompt_emphasis
from utils.safety.minor_guard import require_allowed

_COUNT_RE = re.compile(
    r"\b(exactly\s+\d+|\d+\s+(people|persons|person|characters?|girls?|boys?|coins?|candles?|windows?))\b"
)


@dataclass
class PromptExtras:
    """Negatives and flags gathered while composing the prompt, consumed by the prompt stack."""

    character_negative: str = ""
    scene_negative: str = ""
    scale_distortion: bool = False


@dataclass
class PromptPlan:
    """Final text for the encoders plus what later phases read from the prompt."""

    prompt: str
    negative: str
    emphasis_segments: list = field(default_factory=list)
    num_gen: int = 1
    expected_texts: list = field(default_factory=list)
    photo_grade: str = "none"
    photo_filter: str = "none"
    photo_grain: str = "none"
    photo_pack: str = "none"
    is_photo_prompt: bool = False
    auto_pick_metric: str = ""

    @property
    def expected_ocr(self) -> str:
        return str(self.expected_texts[0]) if self.expected_texts else ""

    @property
    def has_count(self) -> bool:
        return bool(_COUNT_RE.search(self.prompt.lower()))


def _join(*parts: str, sep: str = ", ") -> str:
    return sep.join(p for p in parts if p)


def load_box_layout(args: Any, prompt: str) -> str:
    """Load ``--box-layout`` into ``args._box_layout_spec``; returns the prompt with its global text merged."""
    try:
        from utils.generation.regional_box_prompting import layout_text_from_regions, load_box_layout_file

        spec = load_box_layout_file(str(args.box_layout).strip())
    except Exception as e:
        print(f"Warning: --box-layout failed: {e}", file=sys.stderr)
        return prompt
    args._box_layout_spec = spec
    layout_line = layout_text_from_regions(spec)
    if spec.global_prompt and spec.global_prompt.lower() not in prompt.lower():
        prompt = _join(spec.global_prompt, prompt)
    if not prompt:
        prompt = layout_line
    elif str(getattr(args, "box_layout_mode", "regional_cfg") or "regional_cfg").lower() == "text_only":
        prompt = f"{prompt}. {layout_line}"
    if spec.global_negative and not getattr(args, "negative_prompt", "").strip():
        args.negative_prompt = spec.global_negative
    print(f"Box layout: {len(spec.regions)} region(s) ({', '.join(r.name for r in spec.regions)})", file=sys.stderr)
    return prompt


def compose_user_prompt(args: Any, parser: Any) -> None:
    """Build the user's prompt from --prompt / --prompt-file / --tags / --lora-trigger / layouts; screen it."""
    has_tags = bool(getattr(args, "tags", "").strip() or getattr(args, "tags_file", "").strip())
    has_prompt_file = bool(getattr(args, "prompt_file", "").strip())
    has_prompt_layout = bool(getattr(args, "prompt_layout", "").strip())
    has_cast = bool(str(getattr(args, "cast_json", "") or "").strip())
    if not (args.prompt or has_tags or has_prompt_file or has_prompt_layout or args.box_layout.strip() or has_cast):
        parser.error(
            "Provide at least one of --prompt, --prompt-file, --tags, --tags-file, --prompt-layout, --box-layout "
            "or --cast-json"
        )

    prompt = (args.prompt or "").strip()
    if has_prompt_file:
        pf_path = Path(args.prompt_file)
        if pf_path.exists():
            prompt = pf_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not prompt:
                print(f"Warning: --prompt-file is empty: {pf_path}", file=sys.stderr)
    args._raw_prompt_before_compose = prompt
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()]
    if getattr(args, "tags_file", "").strip():
        tags_path = Path(args.tags_file)
        if tags_path.exists():
            raw = tags_path.read_text(encoding="utf-8", errors="ignore")
            tags += [t.strip() for line in raw.strip().split("\n") for t in line.split(",") if t.strip()]
        else:
            print(f"Warning: --tags-file not found: {tags_path}", file=sys.stderr)
    if tags:
        from data.caption_utils import prompt_from_tags

        prompt = _join(prompt_from_tags(tags), prompt)
    if getattr(args, "lora_trigger", "").strip() and args.lora:
        prompt = _join(args.lora_trigger.strip(), prompt)
    # A cast can supply the box layout, character sheets and moodboard, so it runs before they are read.
    args.prompt = prompt
    _apply_cast_and_corpus(args)
    prompt = str(args.prompt or "")

    args._prompt_layout_negative = ""
    args._used_prompt_layout = False
    args._layout_compiled = None
    args._box_layout_spec = None
    args._regional_cfg_plan = None
    if args.box_layout.strip():
        prompt = load_box_layout(args, prompt)
    if has_prompt_layout:
        try:
            from utils.prompt.prompt_layout import load_prompt_layout_file, merge_prompt_with_layout

            compiled = load_prompt_layout_file(str(args.prompt_layout).strip())
            args._prompt_layout_negative = compiled.negative or ""
            args._used_prompt_layout = True
            args._layout_compiled = compiled
            prompt = merge_prompt_with_layout(compiled.positive, prompt, layout_first=True)
        except Exception as e:
            print(f"Warning: --prompt-layout failed: {e}", file=sys.stderr)

    if getattr(args, "subject_first", False) and prompt and not args._used_prompt_layout:
        from data.caption_utils import prompt_from_tags

        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        if len(parts) > 1:
            prompt = prompt_from_tags(parts)
    if prompt:
        args.prompt = prompt
    require_allowed(str(args.prompt or ""), str(getattr(args, "negative_prompt", "") or ""))
    # The prompt stack's token budget never drops these; everything added later competes for the rest.
    args._user_prompt = str(args.prompt or "")
    args._user_negative = str(getattr(args, "negative_prompt", "") or "")
    _expand_artist_mentions(args)


def _apply_cast_and_corpus(args: Any) -> None:
    """``--cast-json`` (multi-character cast) and ``--corpus-jsonl`` (retrieved captions + moodboard)."""
    try:
        from utils.generation.inventions.wire import apply_cast_and_corpus_to_namespace

        apply_cast_and_corpus_to_namespace(args, work_dir="outputs/_invention")
    except Exception as e:
        print(f"Warning: --cast-json / --corpus-jsonl skipped: {e}", file=sys.stderr)


def _expand_artist_mentions(args: Any) -> None:
    """``@artist`` style mentions and ``+category`` blocks (characters, buildings, vehicles, ...)."""
    raw = str(getattr(args, "prompt", "") or "")
    if not raw:
        return
    strength = float(getattr(args, "artist_strength", 1.0) or 1.0)
    if getattr(args, "prompt_compose", True):
        if "@" not in raw and "+" not in raw:
            return
        from utils.prompt.prompt_composer import compose_prompt

        idx = str(getattr(args, "artist_index", "") or "").strip() or None
        composed = compose_prompt(raw, artist_strength=strength, artist_index=idx)
        args.prompt = composed.positive
        if composed.artists:
            print(f"Artist style: {', '.join(composed.artists)}")
        if composed.blocks:
            print(f"Prompt blocks: {', '.join(sorted(composed.blocks))}")
    elif getattr(args, "artist_style", True) and "@" in raw:
        from utils.prompt.artist_registry import expand_artist_mentions

        args.prompt, artists = expand_artist_mentions(raw, strength=strength)
        if artists:
            print(f"Artist style: {', '.join(artists)}")


def compose_prompt_additions(args: Any) -> PromptExtras:
    """Character, scene and style additions; style packs go after the user's text so the budget can gate them."""
    extras = PromptExtras()
    anatomy = _parse_scale_csv(getattr(args, "anatomy_scale", ""))
    objects = _parse_scale_csv(getattr(args, "object_scale", ""))
    scene = _parse_scale_csv(getattr(args, "scene_scale", ""))
    extras.scale_distortion = bool(anatomy or objects or scene)
    if getattr(args, "gender_swap", False) and getattr(args, "prompt", ""):
        args.prompt = _apply_gender_swap(args.prompt)
    args.prompt = _join(getattr(args, "prompt", "") or "", _build_size_tokens(anatomy, objects, scene))

    auto_oc_negative = _auto_original_character(args)
    char_pos, char_neg = _character_sheets(args)
    char_neg = _join(char_neg, str(getattr(args, "character_negative_extra", "") or "").strip())
    char_neg = _join(char_neg, auto_oc_negative)
    char_pos = _join(char_pos, str(getattr(args, "character_prompt_extra", "") or "").strip())
    extras.character_negative = char_neg
    args.prompt = _join(args.prompt, char_pos)

    if getattr(args, "scene_blueprint", "").strip():
        try:
            from utils.prompt.scene_blueprint import load_scene_blueprint

            scene_pos, extras.scene_negative = load_scene_blueprint(
                args.scene_blueprint, strength=float(getattr(args, "scene_blueprint_strength", 1.0))
            )
            args.prompt = _join(args.prompt, scene_pos)
        except Exception as e:
            print(f"Warning: failed to load --scene-blueprint: {e}", file=sys.stderr)

    args.prompt = _join(args.prompt, *_style_packs(args))

    if getattr(args, "originality", 0.0) and args.prompt:
        try:
            from utils.prompt.originality_augment import inject_originality_tokens

            strength = max(0.0, min(1.0, float(args.originality)))
            rng = np.random.default_rng(int(args.seed) + int(1000 * strength))
            args.prompt = inject_originality_tokens(args.prompt, strength, rng)
            if getattr(args, "creativity", None) is None:
                args.creativity = strength
        except Exception:
            pass
    return extras


def _auto_original_character(args: Any) -> str:
    """Append a synthesised original-character block when the prompt asks for one; returns its negative."""
    if not (getattr(args, "auto_original_character", True) and getattr(args, "prompt", "")):
        return ""
    try:
        from utils.prompt.auto_oc import infer_auto_original_character

        profile = infer_auto_original_character(
            str(args.prompt),
            seed=int(getattr(args, "seed", 0)) + int(getattr(args, "auto_oc_seed_offset", 0)),
            style_context=", ".join(
                str(getattr(args, k, "") or "") for k in ("style", "hard_style", "preset", "op_mode")
            ),
        )
    except Exception:
        return ""
    if profile is None or str(getattr(args, "character_sheet", "") or "").strip():
        return ""
    block = profile.to_prompt_block()
    if block:
        args.prompt = f"{args.prompt}, {block}".strip(", ")
    return str(getattr(profile, "negative_block", "") or "").strip()


def _character_sheets(args: Any) -> tuple[str, str]:
    """(positive, negative) additions from ``--character-sheet`` files, labelled when several share a scene."""
    sheet_paths = [s.strip() for s in str(getattr(args, "character_sheet", "") or "").split(",") if s.strip()]
    pos_all, neg_all = [], []
    for sp in sheet_paths:
        try:
            pos_i, neg_i = _load_character_sheet(
                sp,
                uncensored_mode=bool(getattr(args, "uncensored_mode", False)),
                character_strength=float(getattr(args, "character_strength", 1.0)),
            )
        except Exception as e:
            print(f"Warning: failed to load --character-sheet '{sp}': {e}", file=sys.stderr)
            continue
        if pos_i:
            pos_all.append(pos_i)
        if neg_i:
            neg_all.append(neg_i)
    use_labels = bool(getattr(args, "label_multi_character_sheets", False)) or (
        len(sheet_paths) >= 2 and str(getattr(args, "composition_mode", "none") or "none") == "multi_character"
    )
    if use_labels and len(pos_all) >= 2:
        from utils.prompt.multi_subject import merge_character_sheet_positives, multi_sheet_extra_negatives_csv
        from utils.prompt.tags import merge_csv_unique

        return (
            merge_character_sheet_positives(pos_all),
            _join(merge_csv_unique(*neg_all), multi_sheet_extra_negatives_csv()),
        )
    return ", ".join(pos_all).strip(", "), ", ".join(neg_all).strip(", ")


def _style_packs(args: Any) -> list[str]:
    """Text implied by --hard-style, --naturalize(-deep), --anti-bleed and --diversity."""
    from config.defaults.prompt_domains import (
        CONCEPT_BLEEDING_POSITIVE,
        DIVERSITY_POSITIVE,
        HARD_STYLE_RECOMMENDED_PROMPTS,
        NATURAL_LOOK_POSITIVE,
        NATURAL_LOOK_POSITIVE_DEEP,
    )

    packs = []
    if getattr(args, "hard_style", None):
        packs.append(HARD_STYLE_RECOMMENDED_PROMPTS.get(args.hard_style, [None])[0] or "")
    if getattr(args, "naturalize", False):
        packs.append(NATURAL_LOOK_POSITIVE_DEEP if getattr(args, "naturalize_deep", False) else NATURAL_LOOK_POSITIVE)
    if getattr(args, "anti_bleed", False):
        packs.append(CONCEPT_BLEEDING_POSITIVE)
    if getattr(args, "diversity", False):
        packs.append(DIVERSITY_POSITIVE)
    return packs


def build_prompt_plan(args: Any, m: Any, extras: PromptExtras) -> PromptPlan:
    """Feature, enrichment and prompt-stack passes; returns the text the encoders see."""
    from utils.generation.sample_features import apply_sample_feature_prompt_phase

    apply_sample_feature_prompt_phase(args, steps=int(getattr(args, "steps", 50) or 50))
    if bool(getattr(args, "apply_user_taste", False)):
        from utils.generation.user_taste import apply_taste_to_prompts, load_user_taste

        pos, neg = apply_taste_to_prompts(
            str(getattr(args, "prompt", "") or ""), str(getattr(args, "negative_prompt", "") or ""), load_user_taste()
        )
        args.prompt = pos
        if neg:
            args.negative_prompt = neg
    _apply_invention_stack(args)

    num_gen = max(1, getattr(args, "num", 1))
    if getattr(args, "explore_styles", False) and num_gen < int(getattr(args, "invent_styles", 3) or 3):
        num_gen = int(args.invent_styles)
        args.num = num_gen
    if str(getattr(args, "pick_best", "none") or "none").strip().lower() not in ("none", "") and num_gen < 3:
        num_gen = 3
        args.num = num_gen

    if "(" in args.prompt or "[" in args.prompt:
        prompt, emphasis = parse_prompt_emphasis(args.prompt)
    else:
        prompt, emphasis = args.prompt, []
    if getattr(args, "expand_prompt", False) and prompt.strip() and not getattr(args, "_skip_prompt_expansion", False):
        from utils.superior.inference_pipeline import expand_prompt_heuristic

        prompt = expand_prompt_heuristic(prompt)
        print("Superior: expanded prompt heuristically.", file=sys.stderr)
    hm = str(getattr(args, "human_made", "none") or "none").lower().strip()
    if hm not in ("none", "off", "0", ""):
        from utils.quality.human_made import append_human_made_prompt_fragments

        prompt, args.negative_prompt = append_human_made_prompt_fragments(
            prompt, str(getattr(args, "negative_prompt", "") or ""), hm
        )
    prompt = _ground_and_design(args, prompt, m.device)
    if getattr(args, "boost_quality", False) and prompt.strip():
        from data.caption_utils import QUALITY_PREFIX

        if not prompt.strip().lower().startswith("masterpiece"):
            prompt = f"{QUALITY_PREFIX}{prompt}".strip()
    prompt, num_gen = _multi_instance(args, prompt, num_gen)
    prompt = _detailed_scene(args, prompt)
    prompt = _fix_region(args, prompt, m.image_size)

    expected_texts = _parse_expected_texts(getattr(args, "expected_text", ""))
    if not expected_texts and bool(getattr(args, "auto_expected_text", True)):
        expected_texts = _infer_expected_texts_from_prompt(prompt)
        if expected_texts:
            print(f"Inferred expected text from prompt: {expected_texts}", file=sys.stderr)
    if bool(getattr(args, "auto_constraint_boost", True)) and (expected_texts or _expected_count(args, prompt) > 0):
        if num_gen < 4:
            print(f"Auto constraint boost: num {num_gen} -> 4", file=sys.stderr)
            num_gen = 4
    if getattr(args, "ocr_fix", False) and expected_texts:
        args.text_in_image = True
        prompt = _maybe_append_text_says(prompt, expected_texts)
    args.prompt = prompt

    from utils.prompt.stack import apply_sample_prompt_stack

    prompt, negative = apply_sample_prompt_stack(
        args,
        prompt,
        character_negative_additions=extras.character_negative,
        scene_negative_additions=extras.scene_negative,
        apply_scale_distortion=extras.scale_distortion,
        tokenizer=m.tokenizer,
        max_tokens=TEXT_MAX_TOKENS,
    )
    args.prompt = prompt
    # The stack's photo-realism stage decides these, so they are read only after it ran.
    return PromptPlan(
        prompt=prompt,
        negative=negative,
        emphasis_segments=emphasis,
        num_gen=num_gen,
        expected_texts=expected_texts,
        photo_grade=str(getattr(args, "_pr_grade_ef", "none") or "none"),
        photo_filter=str(getattr(args, "_pr_filter_ef", "none") or "none"),
        photo_grain=str(getattr(args, "_pr_grain_ef", "none") or "none"),
        photo_pack=str(getattr(args, "_pr_pack_ef", "none") or "none"),
        is_photo_prompt=bool(getattr(args, "_is_photo_prompt", False)),
        auto_pick_metric=str(getattr(args, "_auto_pick_metric", "") or ""),
    )


def _apply_invention_stack(args: Any) -> None:
    """``--invention-stack``; a box layout it plans is loaded here because the user's layout was read earlier."""
    before = str(getattr(args, "box_layout", "") or "")
    try:
        from utils.generation.inventions.wire import apply_invention_to_namespace

        apply_invention_to_namespace(args)
    except Exception as e:
        print(f"Warning: invention stack skipped: {e}", file=sys.stderr)
        return
    if str(getattr(args, "box_layout", "") or "") != before and getattr(args, "_box_layout_spec", None) is None:
        args.prompt = load_box_layout(args, str(args.prompt or ""))


def _expected_count(args: Any, prompt: str) -> int:
    """``--expected-count`` or the people / object count the prompt asks for."""
    count = int(getattr(args, "expected_count", 0) or 0)
    if count > 0:
        return count
    from utils.quality.test_time_pick import infer_expected_object_count, infer_expected_people_count

    target = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
    if target == "objects":
        return infer_expected_object_count(prompt)[0]
    count = infer_expected_people_count(prompt)
    if count <= 0 and target == "auto":
        count = infer_expected_object_count(prompt)[0]
    return count


def _ground_and_design(args: Any, prompt: str, device: Any) -> str:
    """Local RAG, agentic facts, composition brief, visual-design packs and creative RAG."""
    local_rag = str(getattr(args, "local_rag_jsonl", "") or "").strip()
    if local_rag:
        try:
            from utils.superior.inference_pipeline import SuperiorPromptStack

            stack = SuperiorPromptStack(rag_jsonl=local_rag, rag_top_k=int(getattr(args, "local_rag_top_k", 8) or 8))
            prompt = stack.enrich(prompt)
            print(f"Local RAG: enriched prompt from {local_rag}", file=sys.stderr)
        except Exception as e:
            print(f"Local RAG skipped: {e}", file=sys.stderr)
    prompt = _agentic_facts(args, prompt)

    brief = str(getattr(args, "composition_brief", "off") or "off")
    if brief != "off" and prompt.strip():
        try:
            from utils.prompt.composition_brief import apply_composition_brief

            prompt = apply_composition_brief(prompt, brief)  # type: ignore[arg-type]
        except Exception as e:
            print(f"Composition brief skipped: {e}", file=sys.stderr)

    args._visual_design_negative = ""
    vd_preset = str(getattr(args, "visual_design_preset", "") or "").strip().lower()
    if vd_preset and prompt.strip():
        try:
            from utils.visual_design import apply_visual_design_preset_to_prompt

            prompt, args.visual_design_domain, args.visual_design_intensity = apply_visual_design_preset_to_prompt(
                prompt, vd_preset
            )
        except Exception as e:
            print(f"Visual design preset skipped: {e}", file=sys.stderr)
    vd_domain = str(getattr(args, "visual_design_domain", "none") or "none").lower().strip()
    if vd_domain != "none" and prompt.strip():
        try:
            from utils.visual_design import apply_visual_design_stage

            vd = apply_visual_design_stage(
                prompt,
                cli_domain=vd_domain,
                intensity=str(getattr(args, "visual_design_intensity", "standard") or "standard"),
                use_negative_pack=bool(getattr(args, "visual_design_negative_pack", False)),
                emit=lambda msg: print(msg, file=sys.stderr),
            )
            prompt = vd.prompt
            if vd.negative_addon:
                args._visual_design_negative = vd.negative_addon
        except Exception as e:
            print(f"Visual design pack skipped: {e}", file=sys.stderr)
    if getattr(args, "creative_rag", False) and prompt.strip():
        prompt = _creative_rag(args, prompt, device)
    return prompt


def _agentic_facts(args: Any, prompt: str) -> str:
    """Merge Gen-Searcher JSON / JSONL facts (``--agentic-facts-json``) into the prompt."""
    facts_path = str(getattr(args, "agentic_facts_json", "") or "").strip()
    if not facts_path:
        return prompt
    try:
        from utils.prompt.rag_prompt import (
            load_facts_from_gen_searcher_json,
            load_facts_from_jsonl,
            merge_facts_into_prompt,
        )

        mode = str(getattr(args, "agentic_facts_format", "auto") or "auto").lower().strip()
        max_facts = max(1, int(getattr(args, "agentic_max_facts", 16) or 16))
        facts: list = []
        if mode in ("auto", "gen_searcher"):
            facts = load_facts_from_gen_searcher_json(facts_path, max_entries=max_facts)
        if not facts and mode in ("auto", "jsonl_text"):
            facts = load_facts_from_jsonl(facts_path, max_entries=max_facts)
        if not facts:
            print(f"Agentic grounding: no facts found in {facts_path}", file=sys.stderr)
            return prompt
        max_chars = max(256, int(getattr(args, "agentic_facts_max_chars", 2400) or 2400))
        print(f"Agentic grounding merged {len(facts)} facts from {facts_path}", file=sys.stderr)
        return merge_facts_into_prompt(prompt, facts, max_chars=max_chars)
    except Exception as e:
        print(f"Warning: agentic fact grounding failed: {e}", file=sys.stderr)
        return prompt


def _creative_rag(args: Any, prompt: str, device: Any) -> str:
    """Semantic conflict resolution plus intent-aware enrichment (moondream2 / Qwen2.5 when available)."""
    try:
        from utils.prompt.advanced_prompting import (
            CreativeRAGOptimizer,
            classify_prompt_intent,
            resolve_semantic_conflicts,
        )

        level = float(getattr(args, "creative_rag_level", 0.7) or 0.7)
        refs = [str(getattr(args, "creative_rag_image", "") or "").strip()]
        refs += str(getattr(args, "creative_rag_images", "") or "").split(",")
        refs = list(dict.fromkeys(r.strip() for r in refs if r.strip()))[:16]
        if bool(getattr(args, "creative_rag_resolve_conflicts", True)):
            resolved, removed = resolve_semantic_conflicts(prompt)
            if removed:
                print(
                    f"Creative RAG: resolved {len(removed)} semantic contradiction(s): "
                    + ", ".join(f"'{r}'" for r in removed[:4]),
                    file=sys.stderr,
                )
                prompt = resolved
        intent = classify_prompt_intent(prompt)
        print(f"Creative RAG: detected intent={intent}, creativity_level={level:.2f}", file=sys.stderr)
        if refs:
            print(
                f"Creative RAG: reference images={len(refs)} (facts/dissection up to 16; moondream up to 8)",
                file=sys.stderr,
            )
        result = CreativeRAGOptimizer(device=str(device)).optimize(
            prompt,
            reference_image_paths=refs or None,
            creativity_level=level,
            optimization_level="balanced",
            seed=int(getattr(args, "seed", 42)),
            use_rag=True,
        )
        enriched = str(result.get("optimized_prompt", "") or "")
        if enriched.strip():
            added = len(enriched.split(",")) - len(prompt.split(","))
            fallback = getattr(result.get("rag_result"), "fallback_used", True)
            print(
                f"Creative RAG ({'semantic fallback' if fallback else 'Qwen2.5 synthesis'}): "
                f"added {max(0, added)} enrichment(s) [intent={intent}]",
                file=sys.stderr,
            )
            prompt = enriched
    except Exception as e:
        print(f"Creative RAG failed (non-fatal): {e}", file=sys.stderr)
    return prompt


def _multi_instance(args: Any, prompt: str, num_gen: int) -> tuple[str, int]:
    """``--multi-instance-preset`` (+ ``--multi-instance-auto``) for crowds of near-identical subjects."""
    args._multi_instance_negative = ""
    preset = str(getattr(args, "multi_instance_preset", "none") or "none").lower().strip()
    if preset in ("none", "") or not prompt.strip():
        return prompt, num_gen
    try:
        from utils.prompt.multi_instance_scene import apply_multi_instance_preset, describe_limitation_short

        user_n = max(0, int(getattr(args, "multi_instance_count", 0) or 0))
        prompt, neg, min_n, expected, note = apply_multi_instance_preset(prompt, preset, user_expected_count=user_n)
        if neg:
            args._multi_instance_negative = neg
        if min_n > num_gen:
            num_gen = min_n
            args.num = num_gen
            print(
                f"Multi-instance preset '{preset}': raised --num to {num_gen} for broader search "
                f"({describe_limitation_short()})",
                file=sys.stderr,
            )
        if user_n > 0:
            args.expected_count = user_n
        elif expected is not None and expected > 0 and int(getattr(args, "expected_count", 0) or 0) <= 0:
            args.expected_count = int(expected)
        if note:
            print(f"Multi-instance: {note}", file=sys.stderr)
    except Exception as e:
        print(f"Multi-instance preset skipped: {e}", file=sys.stderr)
    if not bool(getattr(args, "multi_instance_auto", False)):
        return prompt, num_gen
    try:
        from utils.prompt.composition_brief import apply_composition_brief
        from utils.prompt.multi_instance_scene import multi_instance_auto_settings, print_multi_instance_hints

        st = multi_instance_auto_settings(preset)
        if st.get("composition_brief_boost") and str(getattr(args, "composition_brief", "off")).lower() == "off":
            args.composition_brief = "auto"
            prompt = apply_composition_brief(prompt, "auto")
            print("Multi-instance auto: composition-brief -> auto", file=sys.stderr)
        if str(getattr(args, "pick_best", "none") or "none").strip().lower() in ("none", "") and st.get("pick_best"):
            args.pick_best = str(st["pick_best"])
            print(f"Multi-instance auto: pick-best -> {st['pick_best']}", file=sys.stderr)
        target = str(getattr(args, "expected_count_target", "auto") or "auto").lower().strip()
        if (
            int(getattr(args, "multi_instance_count", 0) or 0) > 0
            and target == "auto"
            and st.get("expected_count_target")
        ):
            args.expected_count_target = str(st["expected_count_target"])
            print(f"Multi-instance auto: expected-count-target -> {st['expected_count_target']}", file=sys.stderr)
        if str(getattr(args, "pick_best", "none") or "").strip().lower() not in ("none", "") and num_gen < 2:
            num_gen = 2
            args.num = num_gen
            print("Multi-instance auto: --num raised to >= 2 because pick-best is active", file=sys.stderr)
        print_multi_instance_hints(preset)
    except Exception as e:
        print(f"Multi-instance auto skipped: {e}", file=sys.stderr)
    return prompt, num_gen


def _detailed_scene(args: Any, prompt: str) -> str:
    """``--detailed-scene-boost``: restate the key clauses of long scene prompts."""
    args._detailed_scene_negative = ""
    mode = str(getattr(args, "detailed_scene_boost", "auto") or "auto")
    strength = str(getattr(args, "detailed_scene_strength", "lite") or "lite")
    if mode == "off" or not prompt.strip():
        return prompt
    try:
        from utils.prompt.detailed_scene_entities import (
            apply_detailed_scene_boost,
            detailed_scene_warrants_boost,
            extract_key_segments,
        )

        if mode == "on" or (mode == "auto" and detailed_scene_warrants_boost(prompt)):
            prompt, neg = apply_detailed_scene_boost(prompt, "on", strength=strength)  # type: ignore[arg-type]
            if neg:
                args._detailed_scene_negative = neg
            segs = extract_key_segments(prompt)
            if segs:
                print(
                    f"Detailed-scene boost ({strength}): {len(segs)} key clauses (e.g. {segs[:3]!r})", file=sys.stderr
                )
    except Exception as e:
        print(f"Detailed-scene boost skipped: {e}", file=sys.stderr)
    return prompt


def _fix_region(args: Any, prompt: str, image_size: int) -> str:
    """``--fix-region NAME`` with ``--init-image`` + ``--box-layout``: inpaint that box with its own prompt."""
    region = str(getattr(args, "fix_region", "") or "").strip()
    box_spec = getattr(args, "_box_layout_spec", None)
    if not (region and box_spec is not None and args.init_image and not args.init_latent and not args.mask):
        return prompt
    try:
        import tempfile

        from utils.generation.sample_features import write_fix_region_mask

        mask_path, region_prompt = write_fix_region_mask(
            box_spec,
            region,
            image_size=image_size,
            out_path=Path(tempfile.gettempdir()) / f"sdx_fix_{region}.png",
        )
        args.mask = str(mask_path)
        args.inpaint_mode = "mdm"
        print(f"Fix-region: inpainting {region!r} via {args.mask}", file=sys.stderr)
        return _join(prompt, region_prompt)
    except Exception as e:
        print(f"Warning: --fix-region failed: {e}", file=sys.stderr)
        return prompt


__all__ = [
    "PromptExtras",
    "PromptPlan",
    "build_prompt_plan",
    "compose_prompt_additions",
    "compose_user_prompt",
    "load_box_layout",
]
