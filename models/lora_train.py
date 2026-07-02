"""Trainable LoRA / DoRA injection for DiT fine-tuning.

Wraps ``nn.Linear`` layers with low-rank trainable adapters and freezes the base
weights, so a run optimizes only the adapters (a few MB) instead of the full
model. Saved adapters use the same key convention the inference loader in
``models/lora.py`` reads (``<path>.lora_down.weight`` / ``.lora_up.weight`` /
``.alpha`` / ``.dora_magnitude_vector``), so a trained adapter drops straight
into ``sample.py --lora``.

DoRA (weight-decomposed LoRA) adds a per-output-channel magnitude vector on top
of the low-rank update, which usually tracks full fine-tuning more closely at
the same rank.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Default injection targets: attention projections and MLP layers, matched by
# substring against each Linear's dotted module path. Covers the DiT blocks
# without touching embedders/heads.
DEFAULT_TARGETS: Tuple[str, ...] = ("qkv", "q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2", "mlp")


class TrainableLoRALinear(nn.Module):
    """``nn.Linear`` + a single trainable LoRA/DoRA adapter. Base weight frozen."""

    def __init__(self, linear: nn.Linear, *, rank: int, alpha: float, use_dora: bool = False):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        in_f = linear.in_features
        out_f = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.use_dora = bool(use_dora)

        dev = linear.weight.device
        dtype = linear.weight.dtype
        # A (down): rank x in, B (up): out x rank. B starts at zero so the
        # adapter is an identity at step 0 (standard LoRA init).
        self.lora_down = nn.Parameter(torch.empty(self.rank, in_f, device=dev, dtype=dtype))
        self.lora_up = nn.Parameter(torch.zeros(out_f, self.rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))

        if self.use_dora:
            # Magnitude of each output column of the (frozen) base weight.
            base_norm = torch.linalg.norm(linear.weight.detach(), dim=1)
            self.dora_magnitude = nn.Parameter(base_norm.to(device=dev, dtype=dtype))
        else:
            self.register_parameter("dora_magnitude", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_dora:
            base = self.linear(x)
            delta = (x @ self.lora_down.T @ self.lora_up.T) * self.scaling
            return base + delta

        # DoRA: renormalize (W + dW) to the learned per-column magnitude.
        w = self.linear.weight
        dw = (self.lora_up @ self.lora_down) * self.scaling
        merged = w + dw
        norm = torch.linalg.norm(merged, dim=1).clamp_min(1e-6)
        directional = merged / norm.unsqueeze(1)
        eff_w = self.dora_magnitude.unsqueeze(1) * directional
        return nn.functional.linear(x, eff_w, self.linear.bias)

    def adapter_state(self, base_key: str) -> Dict[str, torch.Tensor]:
        """Adapter tensors keyed for the inference loader in models/lora.py."""
        out = {
            f"{base_key}.lora_down.weight": self.lora_down.detach().cpu(),
            f"{base_key}.lora_up.weight": self.lora_up.detach().cpu(),
            f"{base_key}.alpha": torch.tensor(float(self.alpha)),
        }
        if self.use_dora and self.dora_magnitude is not None:
            out[f"{base_key}.dora_magnitude_vector"] = self.dora_magnitude.detach().cpu()
        return out


def _iter_target_linears(
    model: nn.Module, targets: Sequence[str]
) -> List[Tuple[nn.Module, str, str, nn.Linear]]:
    """Yield (parent_module, attr_name, dotted_path, linear) for matching Linears."""
    found: List[Tuple[nn.Module, str, str, nn.Linear]] = []
    module_by_path = dict(model.named_modules())
    for path, module in module_by_path.items():
        if not isinstance(module, nn.Linear):
            continue
        if targets and not any(t in path for t in targets):
            continue
        parent_path, _, attr = path.rpartition(".")
        parent = module_by_path.get(parent_path, model)
        found.append((parent, attr, path, module))
    return found


def inject_trainable_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    use_dora: bool = False,
    targets: Optional[Sequence[str]] = None,
) -> Tuple[List[nn.Parameter], int]:
    """Freeze ``model`` and wrap target Linear layers with trainable adapters.

    Returns ``(trainable_params, num_layers_wrapped)``.
    """
    targets = tuple(targets) if targets is not None else DEFAULT_TARGETS

    # Freeze everything first; adapters below re-enable grad on their own params.
    for p in model.parameters():
        p.requires_grad = False

    wrapped = 0
    for parent, attr, path, linear in _iter_target_linears(model, targets):
        adapter = TrainableLoRALinear(linear, rank=rank, alpha=alpha, use_dora=use_dora)
        setattr(parent, attr, adapter)
        wrapped += 1

    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable, wrapped


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Collect adapter-only tensors from all injected layers, keyed for inference."""
    out: Dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        if isinstance(module, TrainableLoRALinear):
            # ``path`` ends at the wrapper; the inference loader resolves the same
            # dotted path to this module and reads its adapter tensors.
            out.update(module.adapter_state(path))
    return out
