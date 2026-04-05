"""LoRA / DoRA / LyCORIS adapter loading for DiT-style Linear layers.

This module supports common adapter checkpoint layouts:
- LoRA classic: ``*.lora_down.weight`` + ``*.lora_up.weight``
- PEFT-style: ``*.lora_A.weight`` + ``*.lora_B.weight``
- Optional alpha: ``*.alpha`` or ``*.lora_alpha``
- Optional DoRA magnitude vector: ``*.dora_magnitude_vector`` / ``*.dora_scale``

For multi-style prompting with many adapters, optional scale normalization prevents
style blowout when users stack many LoRAs at high strengths.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

_DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")
_ALPHA_SUFFIXES = (".alpha", ".lora_alpha")
_DORA_SUFFIXES = (".dora_magnitude_vector", ".dora_scale")
_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)(?:blocks|layers|transformer_blocks|resblocks|h|encoder_blocks|decoder_blocks)\.(\d+)(?:\.|$)"
)


def _get_lora_state_dict(path_or_state: Union[str, Path, Dict]) -> Dict[str, torch.Tensor]:
    if isinstance(path_or_state, dict):
        return path_or_state
    path = Path(path_or_state) if not isinstance(path_or_state, str) else Path(path_or_state)
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        except ImportError:
            raise ImportError("LoRA is .safetensors; install safetensors: pip install safetensors")
    return torch.load(path_or_state, map_location="cpu", weights_only=True)


def _strip_prefix_if_present(k: str, prefix: str) -> str:
    if not prefix:
        return k
    return k.replace(prefix, "", 1) if k.startswith(prefix) else k


def _get_attr(obj, name):
    if name.isdigit():
        return obj[int(name)]
    return getattr(obj, name)


def _set_attr(obj, name, value):
    if name.isdigit():
        obj[int(name)] = value
    else:
        setattr(obj, name, value)


def _resolve_module(model: nn.Module, base_key: str) -> Tuple[Optional[nn.Module], Optional[str], Optional[nn.Module]]:
    parts = base_key.split(".")
    parent = model
    for p in parts[:-1]:
        try:
            parent = _get_attr(parent, p)
        except Exception:
            return None, None, None
    leaf = parts[-1]
    try:
        mod = _get_attr(parent, leaf)
    except Exception:
        return None, None, None
    return parent, leaf, mod


def _extract_layer_index(base_key: str) -> Optional[int]:
    m = _LAYER_INDEX_RE.search(base_key)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _stage_bucket(idx: Optional[int], min_idx: int, max_idx: int) -> int:
    # 0=early, 1=mid, 2=late
    if idx is None or min_idx >= max_idx:
        return 1
    pos = float(idx - min_idx) / float(max(1, max_idx - min_idx))
    if pos <= (1.0 / 3.0):
        return 0
    if pos <= (2.0 / 3.0):
        return 1
    return 2


def _policy_stage_weights(policy: str) -> Dict[str, Tuple[float, float, float]]:
    p = str(policy or "off").strip().lower()
    if p == "character_focus":
        return {
            "character": (1.15, 1.00, 0.85),
            "style": (0.90, 1.00, 1.10),
            "detail": (0.85, 1.00, 1.20),
            "composition": (1.05, 1.00, 0.95),
            "other": (1.00, 1.00, 1.00),
        }
    if p == "style_focus":
        return {
            "character": (0.90, 1.00, 1.05),
            "style": (0.90, 1.00, 1.20),
            "detail": (0.85, 1.00, 1.25),
            "composition": (1.00, 1.00, 1.10),
            "other": (1.00, 1.00, 1.00),
        }
    if p == "balanced":
        return {
            "character": (1.05, 1.00, 0.95),
            "style": (0.95, 1.00, 1.05),
            "detail": (0.95, 1.00, 1.10),
            "composition": (1.02, 1.00, 0.98),
            "other": (1.00, 1.00, 1.00),
        }
    return {}


@dataclass
class _AdapterTensors:
    down: Optional[torch.Tensor] = None
    up: Optional[torch.Tensor] = None
    alpha: Optional[float] = None
    dora_mag: Optional[torch.Tensor] = None


class MultiLoRALinear(nn.Module):
    """Linear wrapper supporting multiple LoRA/DoRA/LyCORIS-style adapters."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self._adapters: List[Dict[str, nn.Parameter]] = []
        self._scales: List[float] = []
        self._roles: List[str] = []

    def add_adapter(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        scale: float,
        alpha: Optional[float] = None,
        dora_mag: Optional[torch.Tensor] = None,
        role: str = "style",
    ) -> None:
        rank = int(down.shape[0]) if down.ndim == 2 else 0
        alpha_ratio = float(alpha) / max(1, rank) if alpha is not None else 1.0
        eff_scale = float(scale) * alpha_ratio
        ad = {
            "down": nn.Parameter(down, requires_grad=False),
            "up": nn.Parameter(up, requires_grad=False),
        }
        if dora_mag is not None:
            ad["dora_mag"] = nn.Parameter(dora_mag, requires_grad=False)
        self._adapters.append(ad)
        self._scales.append(eff_scale)
        self._roles.append(str(role or "style").strip().lower())

    def set_scale_normalization(
        self,
        *,
        max_total_scale: float = 1.5,
        role_budgets: Optional[Dict[str, float]] = None,
    ) -> None:
        # Optional per-role caps (e.g. character > style > detail) for multi-style coherence.
        if role_budgets:
            role_to_idx: Dict[str, List[int]] = {}
            for i, r in enumerate(self._roles):
                role_to_idx.setdefault(r, []).append(i)
            for role, idxs in role_to_idx.items():
                if role not in role_budgets:
                    continue
                b = float(role_budgets[role])
                if b <= 0:
                    continue
                s = sum(abs(self._scales[i]) for i in idxs)
                if s > b and s > 1e-8:
                    f = b / s
                    for i in idxs:
                        self._scales[i] = self._scales[i] * f
        total = sum(abs(s) for s in self._scales)
        if total <= float(max_total_scale) or total <= 1e-8:
            return
        fac = float(max_total_scale) / total
        self._scales = [s * fac for s in self._scales]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if not self._adapters:
            return out
        delta = 0.0
        for ad, sc in zip(self._adapters, self._scales):
            d = x @ ad["down"].T @ ad["up"].T
            if "dora_mag" in ad:
                mag = ad["dora_mag"]
                if mag.ndim == 1 and d.shape[-1] == mag.shape[0]:
                    shape = [1] * d.ndim
                    shape[-1] = mag.shape[0]
                    d = d * mag.view(*shape)
            delta = delta + sc * d
        return out + delta


def _extract_adapters(state: Dict[str, torch.Tensor], prefix: str = "") -> Dict[str, _AdapterTensors]:
    out: Dict[str, _AdapterTensors] = {}
    for k, v in state.items():
        k = _strip_prefix_if_present(k, prefix)
        base = None
        kind = None
        for sfx in _DOWN_SUFFIXES:
            if k.endswith(sfx):
                base = k[: -len(sfx)]
                kind = "down"
                break
        if base is None:
            for sfx in _UP_SUFFIXES:
                if k.endswith(sfx):
                    base = k[: -len(sfx)]
                    kind = "up"
                    break
        if base is None:
            for sfx in _ALPHA_SUFFIXES:
                if k.endswith(sfx):
                    base = k[: -len(sfx)]
                    kind = "alpha"
                    break
        if base is None:
            for sfx in _DORA_SUFFIXES:
                if k.endswith(sfx):
                    base = k[: -len(sfx)]
                    kind = "dora"
                    break
        if base is None:
            continue
        out.setdefault(base, _AdapterTensors())
        if kind == "down":
            out[base].down = v
        elif kind == "up":
            out[base].up = v
        elif kind == "alpha":
            try:
                out[base].alpha = float(v.item()) if torch.is_tensor(v) else float(v)
            except Exception:
                pass
        elif kind == "dora":
            out[base].dora_mag = v
    return out


def apply_lora(
    model: nn.Module,
    lora_path_or_state: Union[str, Path, Dict],
    scale: float = 1.0,
    prefix: str = "",
) -> Tuple[nn.Module, int]:
    """Apply one adapter file; supports LoRA/DoRA/LyCORIS-style key variants on Linear layers."""
    return apply_loras(
        model,
        [(lora_path_or_state, scale)],
        prefix=prefix,
        normalize_scales=False,
    )


def apply_loras(
    model: nn.Module,
    lora_specs: List[Union[Tuple[Union[str, Path, Dict], float], Tuple[Union[str, Path, Dict], float, str]]],
    prefix: str = "",
    *,
    normalize_scales: bool = True,
    max_total_scale: float = 1.5,
    role_budgets: Optional[Dict[str, float]] = None,
    stage_policy: str = "auto",
    role_stage_weights: Optional[Dict[str, Tuple[float, float, float]]] = None,
    layer_group: str = "all",
) -> Tuple[nn.Module, int]:
    """
    Apply multiple adapter files with per-adapter scales.

    ``layer_group``: restrict which depth layers receive adapters.
    - ``"all"`` (default) — apply to all layers.
    - ``"first"`` — first third of layers only (structure/layout).
    - ``"middle"`` — middle third (fine detail).
    - ``"last"`` — last third (aesthetics/style).

    This maps to the research finding that early DiT layers handle structure,
    middle layers add detail, and late layers handle aesthetics.
    """
    by_layer: Dict[str, List[Tuple[_AdapterTensors, float, str]]] = {}
    for spec in lora_specs:
        if len(spec) >= 3:
            path_or_state, scale, role = spec[0], spec[1], spec[2]
        else:
            path_or_state, scale, role = spec[0], spec[1], "style"
        state = _get_lora_state_dict(path_or_state)
        ex = _extract_adapters(state, prefix=prefix)
        for base, ad in ex.items():
            if ad.down is None or ad.up is None:
                continue
            by_layer.setdefault(base, []).append((ad, float(scale), str(role or "style").strip().lower()))

    policy = str(stage_policy or "off").strip().lower()
    roles_present = {r for ads in by_layer.values() for _, _, r in ads}
    if policy == "auto":
        policy = "character_focus" if ("character" in roles_present and "style" in roles_present) else "off"
    stage_weights = _policy_stage_weights(policy)
    if role_stage_weights:
        for k, v in role_stage_weights.items():
            try:
                if len(v) >= 3:
                    stage_weights[str(k).strip().lower()] = (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                continue
    known_indices = [i for i in (_extract_layer_index(k) for k in by_layer.keys()) if i is not None]
    min_idx = min(known_indices) if known_indices else 0
    max_idx = max(known_indices) if known_indices else 0

    # Layer-group filtering: restrict adapters to first/middle/last third of depth.
    lg = str(layer_group or "all").strip().lower()
    if lg != "all" and known_indices:
        depth_range = max_idx - min_idx
        if depth_range > 0:
            if lg == "first":
                cutoff_lo, cutoff_hi = min_idx, min_idx + depth_range // 3
            elif lg == "last":
                cutoff_lo, cutoff_hi = min_idx + 2 * (depth_range // 3), max_idx
            else:  # middle
                cutoff_lo = min_idx + depth_range // 3
                cutoff_hi = min_idx + 2 * (depth_range // 3)
            filtered = {}
            for k, v in by_layer.items():
                idx = _extract_layer_index(k)
                if idx is None or cutoff_lo <= idx <= cutoff_hi:
                    filtered[k] = v
            by_layer = filtered

    touched = 0
    for base_key, adapters in by_layer.items():
        parent, leaf, mod = _resolve_module(model, base_key)
        if parent is None or leaf is None or mod is None:
            continue
        if isinstance(mod, nn.Linear):
            wrapper = MultiLoRALinear(mod)
        elif isinstance(mod, MultiLoRALinear):
            wrapper = mod
        else:
            continue

        stage_i = _stage_bucket(_extract_layer_index(base_key), min_idx, max_idx)
        for ad, scale, role in adapters:
            scale_eff = float(scale) * float(stage_weights.get(role, stage_weights.get("other", (1.0, 1.0, 1.0)))[stage_i])
            wrapper.add_adapter(
                ad.down,
                ad.up,
                scale=scale_eff,
                alpha=ad.alpha,
                dora_mag=ad.dora_mag,
                role=role,
            )
        if normalize_scales:
            wrapper.set_scale_normalization(
                max_total_scale=float(max_total_scale),
                role_budgets={str(k).strip().lower(): float(v) for k, v in (role_budgets or {}).items()},
            )
        _set_attr(parent, leaf, wrapper)
        touched += 1

    return model, touched
