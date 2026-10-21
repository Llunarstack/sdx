"""Token budget: the user's words first, the art-direction caption next, then pack text that fits.

Every stage before this one appends comma tags. Left alone, a ten-word prompt reaches a thousand
words and the text encoder keeps only its first ``max_length`` tokens (T5 300, CLIP 77 for the
pooled vector), so whatever the packs put first wins. Long prompts also dilute attention per
token (Attend-and-Excite, Chefer et al. SIGGRAPH 2023). This stage rebuilds the positive as

    [explicit prefixes] + user tags + art-direction caption + pack tags

keeping pack tags in stage order while they fit ``user_tokens + --prompt-budget`` and the encoder
limit, after dropping tags the plan contradicts (``art_direction.gate_tag``) and duplicates. The
user's own tags are never dropped here; if they alone exceed the encoder the encoder truncates
them. The negative keeps the user's negative first, then default and pack negatives in order while
they fit ``--negative-budget``, minus body text for scenes with nobody in them and negations of the
planned look.
"""

from __future__ import annotations

import math
import re
from typing import Any

from utils.prompt.tags import join_tags, split_tags

from ..context import PromptContext

DEFAULT_EXTRA_TOKENS = 96
DEFAULT_NEGATIVE_EXTRA_TOKENS = 160
_PIECE = re.compile(r"\w+|[^\w\s]")


def _encoder_max_tokens(ctx: PromptContext) -> int:
    if ctx.max_tokens:
        return int(ctx.max_tokens)
    try:
        from utils.generation.sample_helpers import TEXT_MAX_TOKENS

        return int(TEXT_MAX_TOKENS)
    except ImportError:
        return 300


def tag_token_counts(tags: list[str], tokenizer: Any = None) -> list[int]:
    """Encoder tokens per tag (no special tokens); 1.35 per word piece without a tokenizer."""
    if not tags:
        return []
    if tokenizer is not None:
        try:
            return [len(ids) for ids in tokenizer(tags, add_special_tokens=False)["input_ids"]]
        except Exception:
            pass
    return [math.ceil(1.35 * len(_PIECE.findall(t))) for t in tags]


def _leading_text(text: str, n_tags: int) -> str:
    """``text`` up to the end of its ``n_tags``-th non-empty comma tag, unchanged."""
    if n_tags <= 0:
        return ""
    seen = 0
    for m in re.finditer(r"[^,]+", text or ""):
        if m.group().strip():
            seen += 1
            if seen == n_tags:
                return text[: m.end()].strip()
    return (text or "").strip()


def _user_keys(text: str) -> set[str]:
    from utils.prompt.prompt_emphasis import parse_prompt_emphasis

    cleaned, _ = parse_prompt_emphasis(text or "")
    return {t.lower() for t in split_tags(cleaned)}


def fit_tags(
    protected: list[str],
    ordered: list[str],
    *,
    limit: int,
    extra: int,
    tokenizer: Any = None,
) -> tuple[list[str], int]:
    """Protected tags plus as many ``ordered`` tags as fit; returns (kept, dropped count)."""
    seen: set[str] = set()
    head: list[str] = []
    for t in protected:
        if t.lower() not in seen:
            seen.add(t.lower())
            head.append(t)
    tail = []
    for t in ordered:
        if t.lower() not in seen:
            seen.add(t.lower())
            tail.append(t)
    head_counts = tag_token_counts(head, tokenizer)
    used = sum(head_counts) + max(0, len(head) - 1) + 1  # commas + EOS
    cap = limit if extra < 0 else min(limit, used + extra)
    kept = list(head)
    dropped = 0
    for tag, n in zip(tail, tag_token_counts(tail, tokenizer), strict=True):
        cost = n + (1 if kept else 0)
        if used + cost <= cap:
            kept.append(tag)
            used += cost
        else:
            dropped += 1
    return kept, dropped


def stage_token_budget(ctx: PromptContext) -> None:
    args = ctx.args
    if args is None:
        return
    from utils.prompt.art_direction import gate_tag

    ad = getattr(args, "_art_direction", None)
    user = str(getattr(args, "_user_prompt", "") or ctx.metadata.get("input_positive", "") or "")
    limit = _encoder_max_tokens(ctx)
    extra = int(getattr(args, "prompt_budget", DEFAULT_EXTRA_TOKENS))
    neg_extra = int(getattr(args, "negative_budget", DEFAULT_NEGATIVE_EXTRA_TOKENS))

    keys = _user_keys(user)
    tags = split_tags(ctx.positive)
    first_user = next((i for i, t in enumerate(tags) if t.lower() in keys), len(tags) if keys else 0)
    head_end = first_user
    while head_end < len(tags) and tags[head_end].lower() in keys:
        head_end += 1
    protected = tags[:head_end] + [t for t in tags[head_end:] if t.lower() in keys]
    additions = [t for t in tags[head_end:] if t.lower() not in keys]
    caption = split_tags(ad.caption()) if ad is not None else []
    gated = [t for t in additions if ad is None or gate_tag(t, ad, user_text=user)]
    kept, dropped = fit_tags(protected, caption + gated, limit=limit, extra=extra, tokenizer=ctx.tokenizer)
    # The leading user text is kept byte-for-byte so emphasis offsets computed on it stay valid.
    head = _leading_text(ctx.positive, head_end)
    head_keys = {t.lower() for t in tags[:head_end]}
    tail = [t for t in kept if t.lower() not in head_keys]
    ctx.positive = ", ".join(p for p in (head, join_tags(tail)) if p)

    user_neg = str(getattr(args, "_user_negative", "") or "")
    neg_keys = _user_keys(user_neg)
    neg_tags = split_tags(ctx.negative)
    neg_protected = [t for t in neg_tags if t.lower() in neg_keys]
    neg_rest = [
        t
        for t in neg_tags
        if t.lower() not in neg_keys and (ad is None or gate_tag(t, ad, user_text=user, negative=True))
    ]
    neg_kept, neg_dropped = fit_tags(neg_protected, neg_rest, limit=limit, extra=neg_extra, tokenizer=ctx.tokenizer)
    ctx.negative = join_tags(neg_kept) or " "

    ctx.metadata["budget"] = {
        "limit": limit,
        "extra": extra,
        "gated": len(additions) - len(gated),
        "dropped": dropped,
        "negative_gated": len(neg_tags) - len(neg_protected) - len(neg_rest),
        "negative_dropped": neg_dropped,
    }
    ctx.trace.append(f"budget:{len(kept)}tags")
    if hasattr(args, "prompt"):
        args.prompt = ctx.positive
