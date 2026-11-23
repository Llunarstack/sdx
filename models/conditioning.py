"""Extra conditioning inputs for the DiTs.

- ControlNet-style control: a control image (edges/depth/pose/layout) becomes spatial features added to
  the DiT patch tokens (``ControlNetEncoder``, ``encode_control_stack``, ``inject_control_tokens``).
- ``ReferenceTokenProjector``: CLIP-style image embeddings -> extra cross-attention tokens (IP-Adapter style).
- ``RAELatentBridge``: Representation-Autoencoder latents <-> SD-style 4-channel DiT latents.
- ``PromptParser`` / ``attribute_binding_heatmap``: rule-based prompt structure (subject, attributes,
  relations, negation, count) for adherence tooling.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTROL_TYPE_NAMES = (
    "unknown",
    "canny",
    "depth",
    "pose",
    "seg",
    "lineart",
    "scribble",
    "normal",
    "hed",
)

_CONTROL_TYPE_ALIASES = {
    "unknown": "unknown",
    "auto": "unknown",
    "none": "unknown",
    "raw": "unknown",
    "canny": "canny",
    "edge": "canny",
    "edges": "canny",
    "depth": "depth",
    "midas": "depth",
    "zoe": "depth",
    "pose": "pose",
    "openpose": "pose",
    "seg": "seg",
    "segment": "seg",
    "segmentation": "seg",
    "lineart": "lineart",
    "line": "lineart",
    "scribble": "scribble",
    "sketch": "scribble",
    "normal": "normal",
    "normalmap": "normal",
    "normals": "normal",
    "normals_map": "normal",
    "hed": "hed",
    "softedge": "hed",
    "soft_edge": "hed",
}
_CONTROL_TYPE_TO_ID = {n: i for i, n in enumerate(CONTROL_TYPE_NAMES)}

# Substring -> type for file paths; first match wins.
_PATH_HINTS = (
    ("pose", "pose"),
    ("canny", "canny"),
    ("edge", "canny"),
    ("depth", "depth"),
    ("midas", "depth"),
    ("zoe", "depth"),
    ("lineart", "lineart"),
    ("scribble", "scribble"),
    ("seg", "seg"),
    ("normal", "normal"),
    ("hed", "hed"),
)


def normalize_control_type_name(value: str) -> str:
    return _CONTROL_TYPE_ALIASES.get(str(value or "").strip().lower(), "unknown")


def control_type_to_id(value) -> int:
    if isinstance(value, int):
        return max(0, min(len(CONTROL_TYPE_NAMES) - 1, value))
    return _CONTROL_TYPE_TO_ID[normalize_control_type_name(value)]


def infer_control_type_from_path(path: str) -> str:
    s = str(path or "").lower()
    return next((norm for key, norm in _PATH_HINTS if key in s), "unknown")


def _as_tensor(value, convert: Callable, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, (list, tuple)):
        return torch.tensor([convert(v) for v in value], device=device, dtype=dtype)
    return torch.tensor(convert(value), device=device, dtype=dtype)


def _broadcast_per_control(t: torch.Tensor, *, batch_size: int, num_controls: int, name: str) -> torch.Tensor:
    """Broadcast a scalar, ``[K]``, ``[B]``, ``[B*K]``, ``[1, K]``, ``[B, 1]`` or ``[B, K]`` value to ``(B, K)``.

    A 1-D value whose length equals both ``B`` and ``K`` is read as per-control.
    """
    b, k = batch_size, num_controls
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        n = t.shape[0]
        if n == k:
            t = t.view(1, k)
        elif n == b:
            t = t.view(b, 1)
        elif n == b * k:
            t = t.view(b, k)
        elif n == 1:
            t = t.view(1, 1)
    if t.ndim != 2 or t.shape[0] not in (1, b) or t.shape[1] not in (1, k):
        raise ValueError(f"{name} shape {tuple(t.shape)} not compatible with batch={b}, controls={k}")
    return t.expand(b, k)


def _ensure_channels(control_image: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Crop extra channels or repeat the last one to reach ``in_channels``."""
    if control_image.ndim != 4:
        raise ValueError(f"control_image must be 4D (B,C,H,W), got {tuple(control_image.shape)}")
    c = int(control_image.shape[1])
    if c >= in_channels:
        return control_image[:, :in_channels]
    extra = control_image[:, -1:].repeat(1, in_channels - c, 1, 1)
    return torch.cat([control_image, extra], dim=1)


def inject_control_tokens(x_tokens: torch.Tensor, control_feat: torch.Tensor, *, patch_tokens: int) -> torch.Tensor:
    """Add control features to the first ``patch_tokens`` tokens; register tokens after them are untouched.

    A token-count mismatch (variable latent size) is resolved by linear interpolation along the token axis.
    """
    p = min(int(patch_tokens), int(x_tokens.shape[1]))
    if p <= 0:
        return x_tokens
    cfeat = control_feat
    if cfeat.shape[1] != p:
        cfeat = F.interpolate(cfeat.transpose(1, 2), size=p, mode="linear", align_corners=False).transpose(1, 2)
    x_patch = x_tokens[:, :p] + cfeat.to(dtype=x_tokens.dtype)
    return torch.cat([x_patch, x_tokens[:, p:]], dim=1)


class ControlNetEncoder(nn.Module):
    """Patchify a control image into DiT-space tokens ``(B, N, D)``, resized to any latent grid."""

    def __init__(
        self,
        control_size: int,
        patch_size: int,
        in_channels: int = 3,
        hidden_size: int = 1152,
        num_control_types: int = 0,
    ):
        super().__init__()
        self.control_size = int(control_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.proj = nn.Conv2d(self.in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.num_control_types = int(max(0, num_control_types))
        self.type_embed = nn.Embedding(self.num_control_types, hidden_size) if self.num_control_types > 0 else None

    def forward(
        self,
        control_image: torch.Tensor,
        *,
        target_hw: tuple[int, int] | None = None,
        control_type=None,
    ) -> torch.Tensor:
        """``control_image``: ``(B, C, H, W)`` with any C (adapted to ``in_channels``).

        ``target_hw`` is the latent H/W before patchify (default ``control_size`` square); the map is resized to
        it, rounded down to a multiple of the patch size.
        """
        if not torch.is_floating_point(control_image):
            control_image = control_image.float()
        control_image = _ensure_channels(control_image, self.in_channels)

        th, tw = target_hw if target_hw is not None else (self.control_size, self.control_size)
        p = self.patch_size
        th = max(p, (int(th) // p) * p)
        tw = max(p, (int(tw) // p) * p)
        if tuple(control_image.shape[2:]) != (th, tw):
            control_image = F.interpolate(control_image, size=(th, tw), mode="bilinear", align_corners=False)

        x = self.norm(self.proj(control_image).flatten(2).transpose(1, 2))
        if self.type_embed is not None and control_type is not None:
            ids = _as_tensor(control_type, control_type_to_id, device=x.device, dtype=torch.long)
            ids = _broadcast_per_control(ids, batch_size=x.shape[0], num_controls=1, name="control_type")[:, 0]
            ids = ids.clamp(0, self.num_control_types - 1)
            x = x + self.type_embed(ids).unsqueeze(1).to(dtype=x.dtype)
        return x


def encode_control_stack(
    encoder: ControlNetEncoder,
    control_image: torch.Tensor,
    *,
    target_hw: tuple[int, int] | None = None,
    control_type=None,
    control_scale=1.0,
) -> torch.Tensor:
    """Encode one ``(B, C, H, W)`` or K stacked ``(B, K, C, H, W)`` controls into fused tokens ``(B, N, D)``.

    ``control_type`` / ``control_scale`` may be a scalar, ``[K]``, ``[B]``, ``[B*K]`` or ``[B, K]``; negative
    scales are clamped to zero.
    """
    if control_image.ndim == 4:
        control_image = control_image.unsqueeze(1)
    elif control_image.ndim != 5:
        raise ValueError(f"control_image must be 4D or 5D, got shape={tuple(control_image.shape)}")

    b, k = int(control_image.shape[0]), int(control_image.shape[1])
    type_flat = None
    if control_type is not None:
        type_ids = _as_tensor(control_type, control_type_to_id, device=control_image.device, dtype=torch.long)
        type_flat = _broadcast_per_control(type_ids, batch_size=b, num_controls=k, name="control_type").reshape(-1)
    feat = encoder(control_image.flatten(0, 1), target_hw=target_hw, control_type=type_flat)
    feat = feat.view(b, k, *feat.shape[1:])
    scale = _as_tensor(control_scale, float, device=feat.device, dtype=feat.dtype)
    scale = _broadcast_per_control(scale, batch_size=b, num_controls=k, name="control_scale").clamp_min(0.0)
    return (feat * scale.view(b, k, 1, 1)).sum(dim=1)


class ReferenceTokenProjector(nn.Module):
    """
    Project a pooled image embedding (B, clip_dim) to (B, num_tokens, hidden_size)
    for concatenation to text_emb inside DiT_Text.
    """

    def __init__(self, clip_dim: int, hidden_size: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        h = max(int(clip_dim), int(hidden_size))
        self.net = nn.Sequential(
            nn.Linear(int(clip_dim), h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, self.hidden_size * self.num_tokens),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[3].weight, std=0.02)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        b = image_embeds.shape[0]
        x = self.net(image_embeds)
        return x.view(b, self.num_tokens, self.hidden_size)


class RAELatentBridge(nn.Module):
    """Maps RAE latent (B, C_rae, h, w) ↔ DiT latent (B, 4, h, w). Spatial size unchanged."""

    def __init__(self, rae_channels: int, dit_channels: int = 4, *, learnable_output_scale: bool = False):
        super().__init__()
        if rae_channels <= 0 or dit_channels <= 0:
            raise ValueError("rae_channels and dit_channels must be positive")
        self.rae_channels = int(rae_channels)
        self.dit_channels = int(dit_channels)
        self.to_dit = nn.Conv2d(self.rae_channels, self.dit_channels, kernel_size=1, bias=True)
        self.to_rae = nn.Conv2d(self.dit_channels, self.rae_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.to_dit.weight)
        nn.init.zeros_(self.to_dit.bias)
        nn.init.xavier_uniform_(self.to_rae.weight)
        nn.init.zeros_(self.to_rae.bias)
        if learnable_output_scale:
            self.scale_dit = nn.Parameter(torch.ones(1))
            self.scale_rae = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("scale_dit", None)
            self.register_parameter("scale_rae", None)

    def rae_to_dit(self, z_rae: torch.Tensor) -> torch.Tensor:
        y = self.to_dit(z_rae)
        if self.scale_dit is not None:
            y = y * self.scale_dit
        return y

    def dit_to_rae(self, z_dit: torch.Tensor) -> torch.Tensor:
        y = self.to_rae(z_dit)
        if self.scale_rae is not None:
            y = y * self.scale_rae
        return y

    def cycle_loss(self, z_rae: torch.Tensor) -> torch.Tensor:
        """L1 reconstruction: z ≈ to_rae(to_dit(z))."""
        return (self.dit_to_rae(self.rae_to_dit(z_rae)) - z_rae).abs().mean()


@dataclass(slots=True)
class SemanticTriple:
    """A (subject, attribute, relation) triple parsed from a prompt."""

    subject: str
    attributes: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    negated: bool = False
    count: int = 1
    token_indices: list[int] = field(default_factory=list)  # word positions of the attributes + subject


@dataclass(slots=True)
class ParsedPrompt:
    triples: list[SemanticTriple]
    raw: str
    negation_indices: list[int] = field(default_factory=list)  # word positions of negation cues
    count_constraints: dict[str, int] = field(default_factory=dict)  # subject -> count


class PromptParser:
    """Rule-based parser for clothing/object subjects, their preceding attributes, negation, counts and
    spatial relations; works on raw words before tokenisation."""

    # Colour, size, material, pattern and condition attributes.
    _ATTR_RE = re.compile(
        "|".join(
            (
                r"\b(red|orange|yellow|green|blue|purple|pink|black|white|gray|grey|brown|golden|silver|dark|light|bright|pale|vivid|neon)\b",
                r"\b(large|small|big|tiny|huge|tall|short|wide|narrow|thick|thin|long)\b",
                r"\b(wooden|metal|leather|silk|cotton|wool|plastic|glass|stone|fabric|denim|lace)\b",
                r"\b(striped|checkered|floral|plain|patterned|solid|transparent|opaque|shiny|matte|glossy)\b",
                r"\b(old|new|worn|torn|clean|dirty|wet|dry|crumpled|pressed|fitted|loose|tight)\b",
            )
        ),
        re.I,
    )
    _SUBJECT_RE = re.compile(
        r"\b(shirt|blouse|dress|skirt|pants|jeans|jacket|coat|hoodie|sweater|vest|suit|tie|scarf|hat|cap|shoes|boots|sneakers|heels|gloves|socks|underwear|bra|bikini|swimsuit|uniform|robe|cloak|armor|helmet"
        r"|bag|backpack|purse|wallet|watch|glasses|sunglasses|necklace|bracelet|ring|earrings|sword|gun|staff|wand|book|phone|laptop|cup|bottle|umbrella|flower|crown|mask|cube|sphere|cone|cylinder|pyramid|box|ball|block|torus)\b",
        re.I,
    )

    # Spatial phrases, longest first: (words, relation on the preceding subject, inverse on the following one)
    _SPATIAL = (
        (("to", "the", "left", "of"), "left", "right"),
        (("to", "the", "right", "of"), "right", "left"),
        (("left", "of"), "left", "right"),
        (("right", "of"), "right", "left"),
        (("on", "top", "of"), "above", "below"),
        (("in", "front", "of"), "in front", "behind"),
        (("next", "to"), "beside", "beside"),
        (("beside",), "beside", "beside"),
        (("above",), "above", "below"),
        (("below",), "below", "above"),
        (("beneath",), "below", "above"),
        (("under",), "below", "above"),
        (("behind",), "behind", "in front"),
    )

    _NEGATION_RE = re.compile(r"\b(no|not|without|lacking|missing|absent)\b")
    _NEGATION_PHRASES = (("devoid", "of"), ("free", "of"))
    _NEGATION_SPAN = 3  # words after a negation cue that count as negated

    _COUNTS = {
        "one": 1,
        "a": 1,
        "an": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def parse(self, prompt: str) -> ParsedPrompt:
        """Parse a prompt string into semantic triples."""
        words = [w.strip(".,;:!?()[]\"'") for w in prompt.lower().split()]

        negation_indices: list[int] = []
        neg_positions: set[int] = set()
        for i, w in enumerate(words):
            cue_len = 2 if tuple(words[i : i + 2]) in self._NEGATION_PHRASES else int(bool(self._NEGATION_RE.match(w)))
            if cue_len:
                negation_indices.append(i)
                neg_positions.update(range(i + cue_len, min(i + cue_len + self._NEGATION_SPAN, len(words))))

        triples: list[SemanticTriple] = []
        count_constraints: dict[str, int] = {}
        for i, word in enumerate(words):
            count = self._COUNTS.get(word)
            if count is not None and i + 1 < len(words):
                count_constraints[words[i + 1]] = count
            if not self._SUBJECT_RE.match(word):
                continue
            # Attributes bind from the 4 words before the subject, unless negated.
            attr_idx = [j for j in range(max(0, i - 4), i) if self._ATTR_RE.match(words[j]) and j not in neg_positions]
            triples.append(
                SemanticTriple(
                    subject=word,
                    attributes=[words[j] for j in attr_idx],
                    negated=i in neg_positions,
                    count=count_constraints.get(word, 1),
                    token_indices=[*attr_idx, i],
                )
            )

        for k, rel, inverse in self._spatial_relations(words):
            before = [t for t in triples if t.token_indices[-1] < k]
            after = [t for t in triples if t.token_indices[-1] > k]
            if before and after:
                before[-1].relations.append(rel)
                after[0].relations.append(inverse)

        return ParsedPrompt(
            triples=triples,
            raw=prompt,
            negation_indices=negation_indices,
            count_constraints=count_constraints,
        )

    def _spatial_relations(self, words: list[str]):
        """Yield ``(start index, relation, inverse)`` per spatial phrase, longest match first."""
        i = 0
        while i < len(words):
            for phrase, rel, inverse in self._SPATIAL:
                if tuple(words[i : i + len(phrase)]) == phrase:
                    yield i, rel, inverse
                    i += len(phrase)
                    break
            else:
                i += 1


def attribute_binding_heatmap(
    attn_weights: torch.Tensor,
    parsed: ParsedPrompt,
    *,
    spatial_h: int,
    spatial_w: int,
) -> torch.Tensor:
    """Head-averaged cross-attention onto the parsed subject/attribute tokens as an ``(H, W)`` map.

    ``attn_weights`` is ``(heads, N, L)`` or ``(B, heads, N, L)`` (first sample used). When ``N`` does not
    match ``spatial_h * spatial_w`` a square grid is assumed; with no parsed tokens the first 16 are used.
    """
    w = attn_weights
    if w.dim() == 4:
        w = w[0]
    if w.dim() != 3:
        raise ValueError(f"expected attn (H,N,L) or (B,H,N,L), got {tuple(attn_weights.shape)}")
    hw = int(spatial_h) * int(spatial_w)
    if w.shape[1] != hw:
        side = int(round(w.shape[1] ** 0.5))
        spatial_h = spatial_w = side
    heat = w.float().mean(dim=0)  # N, L
    token_idx = [int(i) for t in parsed.triples for i in t.token_indices if 0 <= int(i) < heat.shape[1]]
    if not token_idx:
        token_idx = list(range(min(heat.shape[1], 16)))
    sel = heat[:, token_idx].mean(dim=1)
    return sel.reshape(spatial_h, spatial_w)


__all__ = ["ParsedPrompt", "PromptParser", "SemanticTriple", "attribute_binding_heatmap"]
