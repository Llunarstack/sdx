# LoRA (Low-Rank Adaptation) application for DiT.
# Supports multiple LoRAs with per-LoRA scale so styles/concepts blend without slop.
# Loads .safetensors or .pt; optional trigger words for tag-style LoRAs.
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


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


def _find_target_modules(state_dict: Dict[str, torch.Tensor], base_module: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Resolve LoRA key names to actual Linear modules. LoRA keys often look like 'layers.0.self_attn.q_proj.lora_down.weight'."""
    targets = []
    for key in state_dict:
        if "lora_down" not in key and "lora_up" not in key:
            continue
        base_key = key.replace(".lora_down.weight", "").replace(".lora_up.weight", "")
        parts = base_key.split(".")
        obj = base_module
        for p in parts:
            obj = getattr(obj, p)
        if isinstance(obj, nn.Linear):
            targets.append((base_key, obj))
    return list(dict(targets).items())  # unique by base_key


class LoRALinear(nn.Module):
    """Wraps a Linear and adds LoRA: out = linear(x) + scale * (lora_up @ lora_down @ x)."""

    def __init__(self, linear: nn.Linear, lora_down: torch.Tensor, lora_up: torch.Tensor, scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.lora_down = nn.Parameter(lora_down, requires_grad=False)
        self.lora_up = nn.Parameter(lora_up, requires_grad=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = out + self.scale * (x @ self.lora_down.T @ self.lora_up.T)
        return out


def apply_lora(
    model: nn.Module,
    lora_path_or_state: Union[str, Path, Dict],
    scale: float = 1.0,
    prefix: str = "",
) -> Tuple[nn.Module, int]:
    """
    Apply a single LoRA to the model. Modifies Linear layers in place by wrapping with LoRALinear.
    lora_path_or_state: path to .safetensors or .pt, or state dict with keys ending in lora_down.weight / lora_up.weight.
    prefix: optional prefix to strip from keys (e.g. "module." for DDP).
    Returns (model, num_layers_applied).
    """
    state = _get_lora_state_dict(lora_path_or_state)
    # Collect (module_path, lora_down, lora_up)
    lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for k, v in state.items():
        k = k.replace(prefix, "")
        if k.endswith(".lora_down.weight"):
            base = k.replace(".lora_down.weight", "")
            lora_pairs.setdefault(base, (None, None))
            lora_pairs[base] = (v, lora_pairs[base][1])
        elif k.endswith(".lora_up.weight"):
            base = k.replace(".lora_up.weight", "")
            lora_pairs.setdefault(base, (None, None))
            lora_pairs[base] = (lora_pairs[base][0], v)

    def get_attr(obj, name):
        if name.isdigit():
            return obj[int(name)]
        return getattr(obj, name)

    applied = 0
    for base_key, (down, up) in lora_pairs.items():
        if down is None or up is None:
            continue
        parts = base_key.split(".")
        parent = model
        for p in parts[:-1]:
            parent = get_attr(parent, p)
        name = parts[-1]
        try:
            linear = get_attr(parent, name)
        except (AttributeError, IndexError):
            continue
        if not isinstance(linear, nn.Linear):
            continue
        wrapper = LoRALinear(linear, down, up, scale)
        if name.isdigit():
            parent[int(name)] = wrapper
        else:
            setattr(parent, name, wrapper)
        applied += 1
    return model, applied


def apply_loras(
    model: nn.Module,
    lora_specs: List[Tuple[Union[str, Path, Dict], float]],
    prefix: str = "",
) -> Tuple[nn.Module, int]:
    """
    Apply multiple LoRAs with per-LoRA scale. Each spec is (path_or_state, scale).
    LoRAs are applied in order; later LoRAs add to the same layers (so scales blend).
    Returns (model, total_num_layers_applied).
    """
    total = 0
    for path_or_state, scale in lora_specs:
        _, n = apply_lora(model, path_or_state, scale=scale, prefix=prefix)
        total += n
    return model, total
