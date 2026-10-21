"""ControlNet-style conditioning: a control image (edges/depth/pose/layout) becomes spatial features added to
the DiT patch tokens, keeping structure without overpowering text/style."""

from __future__ import annotations

from collections.abc import Callable

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
