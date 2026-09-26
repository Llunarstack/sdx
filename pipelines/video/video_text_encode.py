"""
Video text encode — real prompt → context tensors for VideoDiT (no more zeros).

Uses structured hashing from ``PromptGroundGraph`` so different prompts produce
distinct, adherence-aware embeddings without requiring a heavy T5 download.
Optionally blends CLIP text features when transformers + weights are available.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "encode_prompt_context",
    "encode_prompt_context_batch",
    "null_context",
]


def _hash_token(token: str, dim: int) -> torch.Tensor:
    """Deterministic unit-ish vector from a string (CPU)."""
    h = abs(hash(token.lower().strip())) % (2**31)
    # Expand to dim via LCG
    vals = []
    x = h
    for i in range(dim):
        x = (1103515245 * x + 12345 + i * 9973) % (2**31)
        vals.append((x / 2**31) * 2.0 - 1.0)
    v = torch.tensor(vals, dtype=torch.float32)
    return F.normalize(v, dim=0)


def _structured_sequence(prompt: str, *, dim: int, max_len: int = 32) -> torch.Tensor:
    from .prompt_ground_graph import parse_prompt_ground

    g = parse_prompt_ground(prompt)
    tokens: list[str] = []
    # Priority order: subjects → attributes → counts → actions → camera → raw words
    for e in g.entities:
        if e.negated:
            tokens.append(f"NEG:{e.name}")
            continue
        tokens.append(f"SUBJ:{e.name}:c{e.count}")
        for a in e.attributes:
            tokens.append(f"ATTR:{e.name}:{a}")
        for r in e.relations:
            tokens.append(f"REL:{e.name}:{r}")
    for k, v in g.counts.items():
        tokens.append(f"COUNT:{k}:{v}")
    for a in g.actions:
        tokens.append(f"ACT:{a}")
    if g.camera:
        tokens.append(f"CAM:{g.camera}")
    # Pad with content words from rewritten prompt
    for w in (g.rewritten or prompt or "scene").lower().replace(",", " ").split():
        if len(w) > 2:
            tokens.append(f"W:{w}")
        if len(tokens) >= max_len:
            break
    if not tokens:
        tokens = ["W:scene"]
    tokens = tokens[:max_len]
    seq = torch.stack([_hash_token(t, dim) for t in tokens], dim=0)  # L, D
    # Adherence modulation: boost ATTR/SUBJ, damp NEG
    for i, t in enumerate(tokens):
        if t.startswith("ATTR:") or t.startswith("SUBJ:"):
            seq[i] = seq[i] * 1.12
        elif t.startswith("NEG:"):
            seq[i] = seq[i] * 0.55
        elif t.startswith("COUNT:"):
            seq[i] = seq[i] * 1.08
    return seq.unsqueeze(0)  # 1, L, D


def _try_clip_text(prompt: str, dim: int, device: torch.device) -> torch.Tensor | None:
    try:
        from transformers import CLIPModel, CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        inputs = tok([prompt], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.get_text_features(**inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            elif isinstance(out, torch.Tensor):
                feat = out
            else:
                return None
            feat = F.normalize(feat.float(), dim=-1)
            if feat.shape[-1] != dim:
                # Project with fixed random orthonormal-ish map seeded
                g = torch.Generator(device="cpu")
                g.manual_seed(42)
                proj = torch.randn(feat.shape[-1], dim, generator=g)
                proj = F.normalize(proj, dim=0)
                feat = feat.cpu() @ proj
            return feat.to(device).unsqueeze(1)  # 1, 1, D
    except Exception:
        return None


def encode_prompt_context(
    prompt: str,
    *,
    context_dim: int = 768,
    max_len: int = 32,
    device: str | torch.device = "cpu",
    use_clip: bool = False,
) -> torch.Tensor:
    """
    Return ``(1, L, D)`` context for VideoDiT.

    Always structured; optionally concatenates a CLIP pooled token when available.
    """
    device = torch.device(device)
    seq = _structured_sequence(prompt or "", dim=int(context_dim), max_len=max_len).to(device)
    if use_clip:
        clip_tok = _try_clip_text(prompt or "", int(context_dim), device)
        if clip_tok is not None:
            seq = torch.cat([clip_tok, seq], dim=1)
    # Also apply still-stack modulation semantics on a synthetic embedding copy
    try:
        from utils.generation.quality_stack import modulate_text_for_adherence

        seq = modulate_text_for_adherence(seq, prompt or "")
    except Exception:
        pass
    return seq


def encode_prompt_context_batch(
    prompts: list[str],
    *,
    context_dim: int = 768,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    mats = [encode_prompt_context(p, context_dim=context_dim, device=device) for p in prompts]
    # Pad to max L
    max_l = max(m.shape[1] for m in mats)
    dim = mats[0].shape[2]
    out = []
    for m in mats:
        if m.shape[1] < max_l:
            pad = torch.zeros(1, max_l - m.shape[1], dim, device=m.device)
            m = torch.cat([m, pad], dim=1)
        out.append(m)
    return torch.cat(out, dim=0)


def null_context(context_dim: int = 768, *, max_len: int = 8, device: str | torch.device = "cpu") -> torch.Tensor:
    """CFG-style null / empty prompt embedding."""
    return encode_prompt_context("", context_dim=context_dim, max_len=max_len, device=device)
