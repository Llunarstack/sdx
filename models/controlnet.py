# ControlNet-style conditioning: control image -> spatial features injected into DiT.
# Keeps structure (edges/depth/pose/layout) without overpowering text/style.
from __future__ import annotations

from typing import Optional, Tuple

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
    "hed": "hed",
}
_CONTROL_TYPE_TO_ID = {n: i for i, n in enumerate(CONTROL_TYPE_NAMES)}


def normalize_control_type_name(value: str) -> str:
    s = str(value or "").strip().lower()
    return _CONTROL_TYPE_ALIASES.get(s, "unknown")


def control_type_to_id(value) -> int:
    if value is None:
        return _CONTROL_TYPE_TO_ID["unknown"]
    if isinstance(value, int):
        return int(max(0, min(len(CONTROL_TYPE_NAMES) - 1, value)))
    s = normalize_control_type_name(str(value))
    return _CONTROL_TYPE_TO_ID.get(s, 0)


def infer_control_type_from_path(path: str) -> str:
    s = str(path or "").strip().lower()
    if not s:
        return "unknown"
    for key, norm in (
        ("openpose", "pose"),
        ("pose", "pose"),
        ("canny", "canny"),
        ("edge", "canny"),
        ("depth", "depth"),
        ("midas", "depth"),
        ("zoe", "depth"),
        ("lineart", "lineart"),
        ("scribble", "scribble"),
        ("seg", "seg"),
        ("segment", "seg"),
        ("normal", "normal"),
        ("hed", "hed"),
    ):
        if key in s:
            return norm
    return "unknown"


def _expand_control_type_ids(control_type, *, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if control_type is None:
        return None
    if torch.is_tensor(control_type):
        ids = control_type.to(device=device, dtype=torch.long)
        if ids.ndim == 0:
            ids = ids.view(1)
    elif isinstance(control_type, (list, tuple)):
        ids = torch.tensor([control_type_to_id(x) for x in control_type], device=device, dtype=torch.long)
    else:
        ids = torch.tensor([control_type_to_id(control_type)], device=device, dtype=torch.long)
    if ids.shape[0] == 1 and batch_size > 1:
        ids = ids.expand(batch_size)
    if ids.shape[0] != batch_size:
        raise ValueError(f"control_type batch mismatch: got {ids.shape[0]}, expected {batch_size}")
    return ids.clamp_min(0)


def _expand_control_scale_tensor(
    control_scale,
    *,
    batch_size: int,
    num_controls: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(control_scale):
        cs = control_scale.to(device=device, dtype=dtype)
    elif isinstance(control_scale, (list, tuple)):
        cs = torch.tensor([float(x) for x in control_scale], device=device, dtype=dtype)
    else:
        cs = torch.tensor(float(control_scale), device=device, dtype=dtype)

    if cs.ndim == 0:
        cs = cs.view(1, 1).expand(batch_size, num_controls)
    elif cs.ndim == 1:
        if cs.shape[0] == num_controls:
            cs = cs.view(1, num_controls).expand(batch_size, num_controls)
        elif cs.shape[0] == batch_size:
            cs = cs.view(batch_size, 1).expand(batch_size, num_controls)
        elif cs.shape[0] == batch_size * num_controls:
            cs = cs.view(batch_size, num_controls)
        elif cs.shape[0] == 1:
            cs = cs.view(1, 1).expand(batch_size, num_controls)
        else:
            raise ValueError(
                f"control_scale length {cs.shape[0]} not compatible with batch={batch_size}, controls={num_controls}"
            )
    elif cs.ndim == 2:
        if cs.shape == (batch_size, num_controls):
            pass
        elif cs.shape == (1, num_controls):
            cs = cs.expand(batch_size, num_controls)
        elif cs.shape == (batch_size, 1):
            cs = cs.expand(batch_size, num_controls)
        else:
            raise ValueError(
                f"control_scale shape {tuple(cs.shape)} not compatible with batch={batch_size}, controls={num_controls}"
            )
    else:
        raise ValueError(f"Unsupported control_scale ndim={cs.ndim}")
    return cs.clamp_min(0.0)


def _expand_control_type_stack(
    control_type,
    *,
    batch_size: int,
    num_controls: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if control_type is None:
        return None
    if torch.is_tensor(control_type):
        ids = control_type.to(device=device, dtype=torch.long)
    elif isinstance(control_type, (list, tuple)):
        ids = torch.tensor([control_type_to_id(x) for x in control_type], device=device, dtype=torch.long)
    else:
        ids = torch.tensor([control_type_to_id(control_type)], device=device, dtype=torch.long)

    if ids.ndim == 0:
        ids = ids.view(1, 1).expand(batch_size, num_controls)
    elif ids.ndim == 1:
        if ids.shape[0] == num_controls:
            ids = ids.view(1, num_controls).expand(batch_size, num_controls)
        elif ids.shape[0] == batch_size:
            ids = ids.view(batch_size, 1).expand(batch_size, num_controls)
        elif ids.shape[0] == batch_size * num_controls:
            ids = ids.view(batch_size, num_controls)
        elif ids.shape[0] == 1:
            ids = ids.view(1, 1).expand(batch_size, num_controls)
        else:
            raise ValueError(
                f"control_type length {ids.shape[0]} not compatible with batch={batch_size}, controls={num_controls}"
            )
    elif ids.ndim == 2:
        if ids.shape == (batch_size, num_controls):
            pass
        elif ids.shape == (1, num_controls):
            ids = ids.expand(batch_size, num_controls)
        elif ids.shape == (batch_size, 1):
            ids = ids.expand(batch_size, num_controls)
        else:
            raise ValueError(
                f"control_type shape {tuple(ids.shape)} not compatible with batch={batch_size}, controls={num_controls}"
            )
    else:
        raise ValueError(f"Unsupported control_type ndim={ids.ndim}")
    return ids


def _ensure_channels(control_image: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Convert control channels to expected encoder channels."""
    if control_image.ndim != 4:
        raise ValueError(f"control_image must be 4D (B,C,H,W), got {tuple(control_image.shape)}")
    c = int(control_image.shape[1])
    if c == in_channels:
        return control_image
    if in_channels == 3 and c == 1:
        return control_image.repeat(1, 3, 1, 1)
    if c > in_channels:
        return control_image[:, :in_channels, ...]
    # c < in_channels: repeat last channel(s) to fill.
    rep = in_channels - c
    extra = control_image[:, -1:, ...].repeat(1, rep, 1, 1)
    return torch.cat([control_image, extra], dim=1)


def inject_control_tokens(
    x_tokens: torch.Tensor,
    control_feat: torch.Tensor,
    *,
    control_scale: torch.Tensor | float,
    patch_tokens: int,
) -> torch.Tensor:
    """
    Inject control features into first `patch_tokens` tokens only.
    This preserves register/non-patch tokens while applying robust token-count matching.
    """
    if patch_tokens <= 0:
        return x_tokens
    p = min(int(patch_tokens), int(x_tokens.shape[1]))
    if p <= 0:
        return x_tokens

    x_patch = x_tokens[:, :p, :]
    x_tail = x_tokens[:, p:, :]
    cfeat = control_feat

    if cfeat.shape[1] != p:
        # Token-count mismatch can happen with variable latent size; interpolate token axis.
        cfeat = cfeat.transpose(1, 2)  # (B, D, N)
        cfeat = F.interpolate(cfeat, size=p, mode="linear", align_corners=False)
        cfeat = cfeat.transpose(1, 2)  # (B, p, D)

    if torch.is_tensor(control_scale):
        cs = control_scale.to(device=x_tokens.device, dtype=x_tokens.dtype)
        if cs.ndim == 0:
            cs = cs.view(1, 1, 1)
        elif cs.ndim == 1:
            cs = cs.view(-1, 1, 1)
        elif cs.ndim == 2:
            cs = cs.view(cs.shape[0], cs.shape[1], 1)
    else:
        cs = torch.tensor(float(control_scale), device=x_tokens.device, dtype=x_tokens.dtype).view(1, 1, 1)

    x_patch = x_patch + cs * cfeat.to(dtype=x_patch.dtype)
    if x_tail.numel() == 0:
        return x_patch
    return torch.cat([x_patch, x_tail], dim=1)


class ControlNetEncoder(nn.Module):
    """
    Encodes a control image to DiT-space control tokens (B, N, D).
    Supports dynamic target spatial sizes to match variable latent shapes.
    """

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
        target_hw: Optional[Tuple[int, int]] = None,
        control_type=None,
    ) -> torch.Tensor:
        """
        control_image: (B, C, H, W), C may be 1/3+ (will be adapted to encoder in_channels).
        target_hw: latent-space H/W before patchify; when set, control map resizes to this,
        then gets patchified with this encoder's patch size.
        Returns (B, N, hidden_size), N=(H/p)*(W/p).
        """
        if not torch.is_floating_point(control_image):
            control_image = control_image.float()
        control_image = _ensure_channels(control_image, self.in_channels)

        if target_hw is not None:
            th, tw = int(target_hw[0]), int(target_hw[1])
        else:
            th, tw = self.control_size, self.control_size

        # Ensure divisible by patch size so token grid is stable.
        th = max(self.patch_size, (th // self.patch_size) * self.patch_size)
        tw = max(self.patch_size, (tw // self.patch_size) * self.patch_size)

        h, w = int(control_image.shape[2]), int(control_image.shape[3])
        if h != th or w != tw:
            control_image = F.interpolate(control_image, size=(th, tw), mode="bilinear", align_corners=False)

        x = self.proj(control_image)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if self.type_embed is not None:
            ids = _expand_control_type_ids(control_type, batch_size=x.shape[0], device=x.device)
            if ids is not None:
                ids = ids.clamp_max(self.num_control_types - 1)
                x = x + self.type_embed(ids).unsqueeze(1).to(dtype=x.dtype)
        return x


def encode_control_stack(
    encoder: ControlNetEncoder,
    control_image: torch.Tensor,
    *,
    target_hw: Optional[Tuple[int, int]] = None,
    control_type=None,
    control_scale=1.0,
) -> torch.Tensor:
    """
    Encode single or stacked controls into one fused token tensor (B, N, D).
    control_image: (B,C,H,W) or (B,K,C,H,W)
    control_type/control_scale may be scalar, [K], [B], [B,K], or tensors of same shapes.
    """
    if control_image.ndim == 4:
        b = int(control_image.shape[0])
        feat = encoder(control_image, target_hw=target_hw, control_type=control_type)
        sc = _expand_control_scale_tensor(
            control_scale,
            batch_size=b,
            num_controls=1,
            device=feat.device,
            dtype=feat.dtype,
        )
        return feat * sc[:, 0].view(b, 1, 1)

    if control_image.ndim != 5:
        raise ValueError(f"control_image must be 4D or 5D, got shape={tuple(control_image.shape)}")

    b, k, c, h, w = control_image.shape
    flat = control_image.reshape(b * k, c, h, w)
    type_ids = _expand_control_type_stack(control_type, batch_size=b, num_controls=k, device=control_image.device)
    type_flat = type_ids.reshape(b * k) if type_ids is not None else None
    feat_flat = encoder(flat, target_hw=target_hw, control_type=type_flat)
    d = int(feat_flat.shape[-1])
    feat = feat_flat.view(b, k, -1, d)
    sc = _expand_control_scale_tensor(
        control_scale,
        batch_size=b,
        num_controls=k,
        device=feat.device,
        dtype=feat.dtype,
    ).view(b, k, 1, 1)
    fused = (feat * sc).sum(dim=1)
    return fused
