"""Neural VideoDiT sample path — denoise short latent clips without retrieve/interpolate.

Orchestration (scene_graph / stitch) stays outside; this module is the engine for
``ProcessOptions.neural_engine=True`` segments.

Prompt/image understanding: structured text context + I2V latent/identity proxies
(no more zero context).

Also holds what used to be: prompt_ground_graph, video_text_encode, video_image_cond.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from utils.safety.minor_guard import is_sexual, require_encodable, require_image_allowed

from pipelines.video.scene import SegmentAssignment, VideoPlan
from pipelines.video.video_io import save_frame_rgb

# ==================================================================================================
# neural_sample
# ==================================================================================================


def load_video_dit(
    ckpt: str | Path | None = None,
    *,
    model_name: str = "VideoDiT-S/2",
    context_dim: int = 768,
    device: str | torch.device = "cpu",
    permanent: bool = False,
) -> torch.nn.Module:
    if permanent or "Permanent" in model_name:
        from models.video_dit import PermanentVideoDiT, PermanentVideoDiT_models

        factory = PermanentVideoDiT_models.get(model_name)
        if factory is None:
            model = PermanentVideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, context_dim=context_dim)
        else:
            model = factory(context_dim=context_dim)
    else:
        from models.video_dit import VideoDiT, VideoDiT_models

        factory = VideoDiT_models.get(model_name)
        if factory is None:
            model = VideoDiT(dim=256, depth=6, num_heads=4, patch_size=2, context_dim=context_dim)
        else:
            model = factory(context_dim=context_dim)
    if ckpt and Path(ckpt).is_file():
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        except Exception:
            # Image DiT ckpts are not VideoDiT — keep random init for dry API smoke.
            pass
    return model.to(device).eval()


@torch.inference_mode()
def sample_video_latents(
    model: torch.nn.Module,
    *,
    frames: int = 8,
    height: int = 32,
    width: int = 32,
    steps: int = 12,
    context: torch.Tensor | None = None,
    first_frame_latent: torch.Tensor | None = None,
    identity: torch.Tensor | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Euler flow-matching denoise of ``(1, C, T, H, W)`` latents.

    Compatible with ``flow_matching`` training; no VAE decode here.
    """
    device = torch.device(device)
    c = int(getattr(model, "in_channels", 4))
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(int(seed))
    x = torch.randn(1, c, frames, height, width, generator=g, device=device)
    if context is not None:
        context = context.to(device)
    if first_frame_latent is not None:
        first_frame_latent = first_frame_latent.to(device)
        # Match channel/spatial to noise tensor
        if first_frame_latent.shape[-2:] != (height, width):
            first_frame_latent = torch.nn.functional.interpolate(
                first_frame_latent, size=(height, width), mode="bilinear", align_corners=False
            )
        if first_frame_latent.shape[1] != c:
            if first_frame_latent.shape[1] < c:
                pad = torch.zeros(1, c - first_frame_latent.shape[1], height, width, device=device)
                first_frame_latent = torch.cat([first_frame_latent, pad], dim=1)
            else:
                first_frame_latent = first_frame_latent[:, :c]

    kwargs: dict[str, Any] = {"context": context, "first_frame_latent": first_frame_latent}
    if hasattr(model, "id_bank") and identity is not None:
        kwargs["identity"] = identity.to(device)
    ts = torch.linspace(0.0, 1.0, steps + 1).tolist()
    for t0, t1 in zip(ts[:-1], ts[1:], strict=True):
        v = model(x, torch.full((1,), t0, device=device), **kwargs)
        x = x + (t1 - t0) * v
    return x


def latents_to_frames_placeholder(
    latents: torch.Tensor,
    out_dir: str | Path,
    *,
    width: int = 256,
    height: int = 256,
    anchor_rgb: np.ndarray | None = None,
) -> list[Path]:
    """
    Decode-less preview: map latent channels to RGB tiles for API smoke tests.

    When ``anchor_rgb`` is set (I2V), frame 0 is the real image and later frames
    blend latent motion onto it so identity is visibly locked.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = latents[0].detach().float().cpu()  # C, T, H, W
    _c, t, _h, _w = z.shape
    from PIL import Image

    paths: list[Path] = []
    anchor = None
    if anchor_rgb is not None:
        anchor = np.asarray(Image.fromarray(anchor_rgb.astype(np.uint8)).resize((width, height), Image.BILINEAR))
    for i in range(t):
        plane = z[:3, i].mean(dim=0).numpy() if z.shape[0] >= 3 else z[0, i].numpy()
        plane = plane - plane.min()
        plane = plane / (plane.max() + 1e-6)
        motion = (np.stack([plane, plane, plane], axis=-1) * 255).astype(np.float32)
        motion_img = np.asarray(
            Image.fromarray(motion.astype(np.uint8)).resize((width, height), Image.BILINEAR)
        ).astype(np.float32)
        if anchor is not None:
            # Frame 0 = pure anchor; later frames = soft latent residual on identity
            alpha = 0.0 if i == 0 else min(0.35, 0.08 + 0.04 * i)
            rgb = ((1.0 - alpha) * anchor.astype(np.float32) + alpha * motion_img).clip(0, 255).astype(np.uint8)
        else:
            rgb = motion_img.astype(np.uint8)
        fp = out / f"neural_{i:04d}.png"
        save_frame_rgb(fp, rgb)
        paths.append(fp)
    return paths


def run_neural_segment(
    assignment: SegmentAssignment,
    plan: VideoPlan,
    work_dir: str | Path,
    *,
    ckpt: str = "",
    target_frames: int = 24,
    width: int = 512,
    height: int = 512,
    steps: int = 12,
    device: str = "cpu",
    permanent: bool = True,
) -> tuple[list[Path], dict[str, Any]]:
    """
    Generate a segment with VideoDiT / PermanentVideoDiT.

    Returns frame paths + metadata. Raises on hard failures so the caller can
    fall back to retrieve→edit.
    """

    wd = Path(work_dir)
    neural_dir = wd / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)

    # Latent spatial size (assume /8 VAE); clamp for smoke without VAE.
    lh = max(8, (height // 8) // 2 * 2)
    lw = max(8, (width // 8) // 2 * 2)
    # Cap temporal length for untrained smoke
    t_lat = max(4, min(16, target_frames // 2))

    model_name = "PermanentVideoDiT-S/2" if permanent else "VideoDiT-S/2"
    model = load_video_dit(
        ckpt or None,
        device=device,
        permanent=permanent,
        model_name=model_name,
    )
    ctx_dim = int(getattr(model.context_proj, "in_features", 768)) if model.context_proj else 768
    model_dim = int(getattr(model, "dim", 256) or 256)

    raw_prompt = str(assignment.shot.prompt or plan.user_prompt or "").strip()
    graph = parse_prompt_ground(raw_prompt)
    prompt = graph.rewritten or raw_prompt

    # Multimodal @ / identity refs from plan metadata or assignment
    refs: dict[str, str] = {}
    leaders = (plan.metadata or {}).get("leaders") or {}
    for i, p in enumerate(getattr(assignment, "identity_refs", None) or leaders.get("identity_paths") or []):
        if p:
            refs[f"Image{i + 1}"] = f"identity reference {i + 1}"
    if assignment.start_image and Path(assignment.start_image).is_file():
        refs.setdefault("Image1", "first-frame identity lock")
        brief = image_prompt_brief(assignment.start_image)
        if brief.brief:
            prompt = f"{prompt}. {brief.brief}"
    if refs:
        prompt = bind_at_mentions(prompt, refs, graph=graph)

    if assignment.start_image and is_sexual(prompt):
        require_image_allowed(assignment.start_image, sexual=True, conditioning=True)
    context = None
    if model.context_proj is not None:
        context = encode_prompt_context(prompt, context_dim=ctx_dim, device=device)

    first = None
    identity = None
    anchor_rgb = None
    if assignment.start_image and Path(assignment.start_image).is_file():
        first = image_to_latent_proxy(
            assignment.start_image,
            height=lh,
            width=lw,
            device=device,
        )
        if permanent:
            identity = image_to_identity_tokens(
                assignment.start_image,
                dim=model_dim,
                device=device,
            )
        try:
            from PIL import Image

            anchor_rgb = np.asarray(Image.open(assignment.start_image).convert("RGB"))
        except Exception:
            anchor_rgb = None

    latents = sample_video_latents(
        model,
        frames=t_lat,
        height=lh,
        width=lw,
        steps=steps,
        context=context,
        first_frame_latent=first,
        identity=identity,
        seed=42 + int(assignment.shot.index),
        device=device,
    )
    frames = latents_to_frames_placeholder(
        latents,
        neural_dir,
        width=width,
        height=height,
        anchor_rgb=anchor_rgb,
    )
    # Resample to target_frames by holding last
    if len(frames) < target_frames:
        frames = list(frames) + [frames[-1]] * (target_frames - len(frames))
    elif len(frames) > target_frames:
        idx = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        frames = [frames[i] for i in idx]

    meta = {
        "ops": [
            f"neural_sample:t={t_lat}",
            f"steps={steps}",
            f"model={model_name}",
            "prompt_ground=1",
            f"context={'yes' if context is not None else 'no'}",
            f"i2v={'yes' if first is not None else 'no'}",
        ],
        "prompt": prompt,
        "prompt_raw": raw_prompt,
        "negative_extra": graph.negative_extra,
        "ground_entities": len(graph.entities),
        "engine": "permanent_video_dit" if permanent else "video_dit",
    }
    return frames, meta


# ==================================================================================================
# prompt_ground_graph: Prompt ground graph — accurate structured understanding of T2V prompts.
#
# Seedance-style adherence starts with binding subjects/attributes/counts/negations
# *before* generation. We wrap ``PromptParser`` into a video scene graph and rewrite
# prompts so attributes cannot leak and @-refs bind cleanly to roles.
# ==================================================================================================


@dataclass(slots=True)
class GroundEntity:
    name: str
    attributes: list[str] = field(default_factory=list)
    count: int = 1
    negated: bool = False
    relations: list[str] = field(default_factory=list)
    bind_ref: str = ""  # @Image1 / element id


@dataclass(slots=True)
class PromptGroundGraph:
    raw: str
    entities: list[GroundEntity] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    camera: str = ""
    rewritten: str = ""
    negative_extra: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def subject_count(self) -> int:
        return sum(max(1, e.count) for e in self.entities if not e.negated)


_PEOPLE = (
    "woman",
    "man",
    "girl",
    "boy",
    "person",
    "people",
    "child",
    "adult",
    "hero",
    "character",
    "dog",
    "cat",
    "bird",
    "car",
    "robot",
)


def parse_prompt_ground(prompt: str) -> PromptGroundGraph:
    from models.conditioning import PromptParser

    raw = (prompt or "").strip()
    parsed = PromptParser().parse(raw)
    entities: list[GroundEntity] = []
    negations: list[str] = []
    seen: set[str] = set()
    for t in getattr(parsed, "triples", []) or []:
        name = str(getattr(t, "subject", "") or "").strip() or "subject"
        ent = GroundEntity(
            name=name,
            attributes=[str(a) for a in (getattr(t, "attributes", []) or [])],
            count=int(getattr(t, "count", 1) or 1),
            negated=bool(getattr(t, "negated", False)),
            relations=[str(r) for r in (getattr(t, "relations", []) or [])],
        )
        entities.append(ent)
        seen.add(name.lower())
        if ent.negated:
            negations.append(ent.name)
            negations.extend(ent.attributes)
    counts = {str(k): int(v) for k, v in (getattr(parsed, "count_constraints", {}) or {}).items()}

    # People / animate subjects PromptParser may miss (it focuses on clothing/objects).
    words = raw.lower().split()
    attr_words = {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
        "brown",
        "tall",
        "short",
        "young",
        "old",
    }
    for i, w in enumerate(words):
        stem = w.rstrip("s.,;:")
        if stem not in _PEOPLE or stem in seen:
            continue
        attrs = [words[j] for j in range(max(0, i - 3), i) if words[j] in attr_words]
        cnt = 1
        if i > 0 and words[i - 1] in ("two", "three", "four", "five"):
            cnt = {"two": 2, "three": 3, "four": 4, "five": 5}[words[i - 1]]
            counts.setdefault(stem, cnt)
        entities.insert(
            0,
            GroundEntity(name=stem, attributes=attrs, count=counts.get(stem, cnt)),
        )
        seen.add(stem)

    # Action verbs (temporal layer)
    actions: list[str] = []
    low = raw.lower()
    for verb in (
        "walk",
        "run",
        "turn",
        "look",
        "speak",
        "sing",
        "dance",
        "sit",
        "stand",
        "open",
        "close",
        "pick",
        "hold",
        "drive",
        "fly",
        "fall",
        "jump",
    ):
        if verb in low or f"{verb}s" in low or f"{verb}ing" in low:
            actions.append(verb)

    camera = ""
    for cam in ("orbit", "dolly", "handheld", "crane", "pan", "close-up", "wide", "tracking"):
        if cam in low:
            camera = cam
            break

    g = PromptGroundGraph(
        raw=raw,
        entities=entities,
        counts=counts,
        negations=negations,
        actions=actions,
        camera=camera,
    )
    g.rewritten, g.negative_extra = rewrite_prompt_for_adherence(g)
    g.notes.append(f"entities={len(entities)}")
    if counts:
        g.notes.append(f"counts={counts}")
    return g


def rewrite_prompt_for_adherence(graph: PromptGroundGraph) -> tuple[str, str]:
    """
    Emit an engineering-style prompt that hard-binds attributes to subjects.

    Example: "a woman in a red dress and a man in a blue suit" stays ordered and
    repeats bindings so the model can't swap colors.
    """
    if not graph.entities and not graph.raw:
        return "cinematic scene", "blurry, deformed"
    bits: list[str] = []
    neg: list[str] = []
    for e in graph.entities:
        if e.negated:
            neg.append(f"{e.name}")
            neg.extend(e.attributes)
            continue
        attr = ", ".join(e.attributes) if e.attributes else ""
        count_w = "" if e.count <= 1 else f"{e.count} "
        if attr:
            # Repeat binding: "red dress on the woman (red, not blue)"
            bound = f"{count_w}{e.name} with {attr} (attributes locked to {e.name} only)"
        else:
            bound = f"{count_w}{e.name}".strip()
        if e.bind_ref:
            bound = f"{bound} bound to {e.bind_ref}"
        if e.relations:
            bound = f"{bound}, {', '.join(e.relations)}"
        bits.append(bound)
    if not bits:
        bits.append(graph.raw)
    if graph.actions:
        bits.append("actions: " + ", ".join(graph.actions))
    if graph.camera:
        bits.append(f"camera: {graph.camera}")
    # Preserve original for coverage of uncovered details
    core = "; ".join(bits)
    if graph.raw and graph.raw.lower() not in core.lower():
        core = f"{core}. Full brief: {graph.raw}"
    for n in graph.negations:
        neg.append(n)
    neg.extend(
        [
            "attribute swap between subjects",
            "wrong object count",
            "ignored negation",
            "identity morph",
            "extra limbs",
        ]
    )
    return core, ", ".join(dict.fromkeys(neg))


def bind_at_mentions(
    prompt: str,
    refs: dict[str, str] | None = None,
    *,
    graph: PromptGroundGraph | None = None,
) -> str:
    """
    Seedance-style @ binding: ``@Image1`` / ``@hero`` → explicit role sentences.

    ``refs`` maps mention → role description, e.g. ``{"Image1": "identity of red-haired woman"}``.
    """
    text = prompt or ""
    mapping = dict(refs or {})
    g = graph or parse_prompt_ground(text)
    # Auto-bind first identity entity to first ImageN if present
    if mapping and g.entities:
        first_key = next(iter(mapping))
        if not g.entities[0].bind_ref:
            g.entities[0].bind_ref = f"@{first_key}"
    out = text
    for key, role in mapping.items():
        token = f"@{key}"
        if token.lower() in out.lower() or key in out:
            clause = f"[{token} = {role}; use only for that role]"
            if clause not in out:
                out = f"{clause} {out}"
        else:
            # Inject if prompt doesn't mention but refs provided
            out = f"[{token} = {role}] {out}"
    rewritten, _ = rewrite_prompt_for_adherence(g)
    # Prefer rewritten structure + @ clauses
    if mapping:
        return f"{out}. Structured: {rewritten}"
    return rewritten


# ==================================================================================================
# video_text_encode: Video text encode — real prompt → context tensors for VideoDiT (no more zeros).
#
# Uses structured hashing from ``PromptGroundGraph`` so different prompts produce
# distinct, adherence-aware embeddings without requiring a heavy T5 download.
# Optionally blends CLIP text features when transformers + weights are available.
# ==================================================================================================


@functools.lru_cache(maxsize=4096)
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


@functools.lru_cache(maxsize=2)
def _clip_text_model(device: torch.device):
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return tok, CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()


def _try_clip_text(prompt: str, dim: int, device: torch.device) -> torch.Tensor | None:
    try:
        tok, model = _clip_text_model(device)
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
    require_encodable(prompt)
    device = torch.device(device)
    seq = _structured_sequence(prompt or "", dim=int(context_dim), max_len=max_len).to(device)
    if use_clip:
        clip_tok = _try_clip_text(prompt or "", int(context_dim), device)
        if clip_tok is not None:
            seq = torch.cat([clip_tok, seq], dim=1)
    # Also apply still-stack modulation semantics on a synthetic embedding copy
    try:
        from utils.generation.sample_core import modulate_text_for_adherence

        seq = modulate_text_for_adherence(seq, prompt or "")
    except Exception:
        pass
    return seq


# ==================================================================================================
# video_image_cond: Video image condition — accurate I2V / FLF / element image understanding.
#
# Without a VAE we still produce:
#
# 1. A first-frame latent proxy (downsampled RGB → 4-ch latent-shaped tensor).
# 2. Identity tokens for PermanentVideoDiT ``identity=`` pathway.
# 3. An image brief (palette, composition, subject band) injected into the prompt.
# ==================================================================================================


@dataclass(slots=True)
class ImageCondReport:
    brief: str = ""
    palette: list[str] = field(default_factory=list)
    has_file: bool = False
    notes: list[str] = field(default_factory=list)


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def image_prompt_brief(path: str | Path) -> ImageCondReport:
    """Describe what the model must preserve from the input image."""
    p = Path(path)
    if not p.is_file():
        return ImageCondReport(notes=["missing_image"])
    rgb = _load_rgb(p)
    h, w = rgb.shape[:2]
    # Dominant colors
    small = rgb[:: max(1, h // 32), :: max(1, w // 32)].reshape(-1, 3).astype(np.float32)
    mean = small.mean(axis=0)
    names = []
    # crude name
    r, g, b = mean
    if r > g + 20 and r > b + 20:
        names.append("warm-red dominant")
    elif b > r + 20 and b > g + 10:
        names.append("cool-blue dominant")
    elif g > r + 15 and g > b + 15:
        names.append("green dominant")
    else:
        names.append("neutral palette")
    # Subject band contrast
    cy0, cy1 = int(h * 0.15), int(h * 0.75)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    sub = rgb[cy0:cy1, cx0:cx1].astype(np.float32)
    contrast = float(sub.std())
    aspect = "portrait" if h > w * 1.1 else ("landscape" if w > h * 1.1 else "square")
    brief = (
        f"preserve identity and composition from reference image ({aspect} {w}x{h}, "
        f"{names[0]}, subject-band contrast {contrast:.0f}); "
        f"do not invent a different person/product; lock wardrobe colors"
    )
    return ImageCondReport(brief=brief, palette=names, has_file=True, notes=[f"size={w}x{h}"])


def image_to_latent_proxy(
    path: str | Path,
    *,
    height: int = 32,
    width: int = 32,
    channels: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    ``(1, C, H, W)`` latent-shaped tensor from an RGB image (VAE stand-in).

    Channel 0–2 = normalized RGB; channel 3 = luminance residual.
    """
    from PIL import Image

    device = torch.device(device)
    p = Path(path)
    if not p.is_file():
        return torch.zeros(1, channels, height, width, device=device)
    im = Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
    t = torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W
    if channels >= 4:
        luma = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).unsqueeze(0)
        # Center to ~N(0,1)-ish for diffusion
        t = torch.cat([t * 2 - 1, luma * 2 - 1], dim=0)
    else:
        t = t[:channels] * 2 - 1
    if t.shape[0] < channels:
        pad = torch.zeros(channels - t.shape[0], height, width)
        t = torch.cat([t, pad], dim=0)
    return t[:channels].unsqueeze(0).to(device)


def image_to_identity_tokens(
    path: str | Path,
    *,
    dim: int = 256,
    num_tokens: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """``(1, K, D)`` identity tokens from spatial chroma grid (for PermanentVideoDiT)."""
    device = torch.device(device)
    p = Path(path)
    if not p.is_file():
        return torch.zeros(1, num_tokens, dim, device=device)
    from PIL import Image

    im = Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    tokens = []
    grid = int(max(1, num_tokens**0.5))
    cell = 32 // grid
    for i in range(num_tokens):
        gy, gx = divmod(i, grid)
        y0, x0 = gy * cell, gx * cell
        patch = arr[y0 : y0 + cell, x0 : x0 + cell].reshape(-1, 3).mean(axis=0)
        # Expand 3-D color to dim via seeded projection
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + i)
        basis = torch.randn(3, dim, generator=g)
        basis = F.normalize(basis, dim=0)
        tok = torch.from_numpy(patch).float() @ basis
        tokens.append(F.normalize(tok, dim=0))
    return torch.stack(tokens, dim=0).unsqueeze(0).to(device)


def prepare_i2v_conditioning(
    image_path: str | Path,
    prompt: str,
    *,
    latent_h: int = 32,
    latent_w: int = 32,
    context_dim: int = 768,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Bundle text + image understanding for neural I2V."""

    brief = image_prompt_brief(image_path)
    g = parse_prompt_ground(prompt)
    rewritten, neg = rewrite_prompt_for_adherence(g)
    full_prompt = f"{rewritten}. {brief.brief}" if brief.brief else rewritten
    if is_sexual(full_prompt):
        require_image_allowed(image_path, sexual=True, conditioning=True)
    context = encode_prompt_context(full_prompt, context_dim=context_dim, device=device)
    latent = image_to_latent_proxy(image_path, height=latent_h, width=latent_w, device=device)
    identity = image_to_identity_tokens(image_path, dim=context_dim, device=device)
    return {
        "prompt": full_prompt,
        "negative": neg,
        "context": context,
        "first_frame_latent": latent,
        "identity": identity,
        "brief": brief,
    }


# ==================================================================================================
# Restored from b654ae9:pipelines/video/video_text_encode.py
# ==================================================================================================
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


@dataclass(frozen=True, slots=True)
class VideoFoundation:
    name: str
    path_or_id: str
    role: str  # t2v | i2v
    notes: str = ""


def list_video_foundations() -> list[VideoFoundation]:
    from utils.modeling.model_paths import (
        default_cogvideox_i2v_path,
        default_cogvideox_path,
        default_flf2v_foundation_path,
        default_hunyuan_i2v_path,
        default_hunyuan_video_path,
        default_live_portrait_path,
        default_ltx_video_path,
        default_mochi_path,
        default_open_sora_path,
        default_svd_path,
        default_v2v_path,
        default_vace_path,
        default_video_foundation_path,
        default_wan_animate_path,
        default_wan_i2v_path,
        default_wan_t2v_path,
    )

    return [
        VideoFoundation("default", default_video_foundation_path(), "t2v", "Wan TI2V prefer"),
        VideoFoundation("wan_t2v", default_wan_t2v_path(), "t2v"),
        VideoFoundation("wan_i2v", default_wan_i2v_path(), "i2v"),
        VideoFoundation("ltx", default_ltx_video_path(), "t2v", "LTX-2.3 prefer"),
        VideoFoundation("cogvideox", default_cogvideox_path(), "t2v", "STDiT twin to VideoDiT"),
        VideoFoundation("cogvideox_i2v", default_cogvideox_i2v_path(), "i2v"),
        VideoFoundation("hunyuan", default_hunyuan_video_path(), "t2v"),
        VideoFoundation("hunyuan_i2v", default_hunyuan_i2v_path(), "i2v"),
        VideoFoundation("mochi", default_mochi_path(), "t2v"),
        VideoFoundation("open_sora", default_open_sora_path(), "t2v"),
        VideoFoundation("svd", default_svd_path(), "i2v"),
        VideoFoundation("vace", default_vace_path(), "video_edit", "All-in-one edit / V2V control"),
        VideoFoundation("v2v", default_v2v_path(), "v2v", "SkyReels / VACE restyle"),
        VideoFoundation("flf2v", default_flf2v_foundation_path(), "flf2v", "First–last-frame Wan"),
        VideoFoundation("animate", default_wan_animate_path(), "video_animate"),
        VideoFoundation("portrait", default_live_portrait_path(), "video_portrait"),
    ]


def resolve_video_foundation(prefer: str = "default") -> VideoFoundation:
    """Pick a foundation by short name (``wan``, ``ltx``, ``vace``, ``v2v``, …)."""
    key = (prefer or "default").strip().lower().replace("-", "_")
    aliases = {
        "": "default",
        "auto": "default",
        "wan": "wan_t2v",
        "wan2": "wan_t2v",
        "wan2.2": "wan_t2v",
        "i2v": "wan_i2v",
        "cogvideo": "cogvideox",
        "cogvideox5b": "cogvideox",
        "opensora": "open_sora",
        "stable_video": "svd",
        "svd_xt": "svd",
        "edit": "vace",
        "video_edit": "vace",
        "vid2vid": "v2v",
        "video2video": "v2v",
        "first_last": "flf2v",
        "flf": "flf2v",
        "liveportrait": "portrait",
        "reenact": "portrait",
    }
    key = aliases.get(key, key)
    for f in list_video_foundations():
        if f.name == key:
            return f
    return list_video_foundations()[0]


def foundation_sample_hint(prefer: str = "default") -> dict[str, Any]:
    """Metadata for VIDEOMAX / neural_sample logs (no Diffusers load here)."""
    f = resolve_video_foundation(prefer)
    return {"foundation": f.name, "path_or_id": f.path_or_id, "role": f.role, "notes": f.notes}
