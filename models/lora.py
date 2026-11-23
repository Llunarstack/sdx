"""LoRA / DoRA / LyCORIS adapter loading for DiT-style Linear layers.

Supported checkpoint layouts:
- LoRA classic: ``*.lora_down.weight`` + ``*.lora_up.weight``
- PEFT-style: ``*.lora_A.weight`` + ``*.lora_B.weight``
- Optional alpha: ``*.alpha`` or ``*.lora_alpha``
- Optional DoRA magnitude vector: ``*.dora_magnitude_vector`` / ``*.dora_scale``

Stacking many adapters at high strength blows out the style, so per-layer scales can be normalised to a
total budget (and per-role budgets).

Training: ``inject_trainable_lora`` wraps ``nn.Linear`` layers with trainable LoRA / DoRA adapters
(``TrainableLoRALinear``) and freezes the base weights; ``lora_state_dict`` saves them with the same key
convention the loader reads, so a trained adapter drops straight into ``sample.py --lora``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SUFFIX_KINDS = (
    ("down", (".lora_down.weight", ".lora_A.weight")),
    ("up", (".lora_up.weight", ".lora_B.weight")),
    ("alpha", (".alpha", ".lora_alpha")),
    ("dora_mag", (".dora_magnitude_vector", ".dora_scale")),
)
_LAYER_INDEX_RE = re.compile(
    r"(?:^|\.)(?:blocks|layers|transformer_blocks|resblocks|h|encoder_blocks|decoder_blocks)\.(\d+)(?:\.|$)"
)
_LAYER_GROUPS = {"first": 0, "middle": 1, "last": 2}

# Per-role (early, mid, late) depth multipliers for each stage policy.
_STAGE_POLICIES: dict[str, dict[str, tuple[float, float, float]]] = {
    "character_focus": {
        "character": (1.15, 1.00, 0.85),
        "style": (0.90, 1.00, 1.10),
        "detail": (0.85, 1.00, 1.20),
        "composition": (1.05, 1.00, 0.95),
        "other": (1.00, 1.00, 1.00),
    },
    "style_focus": {
        "character": (0.90, 1.00, 1.05),
        "style": (0.90, 1.00, 1.20),
        "detail": (0.85, 1.00, 1.25),
        "composition": (1.00, 1.00, 1.10),
        "other": (1.00, 1.00, 1.00),
    },
    "balanced": {
        "character": (1.05, 1.00, 0.95),
        "style": (0.95, 1.00, 1.05),
        "detail": (0.95, 1.00, 1.10),
        "composition": (1.02, 1.00, 0.98),
        "other": (1.00, 1.00, 1.00),
    },
}


def _norm_role(role) -> str:
    return str(role or "style").strip().lower()


def _get_lora_state_dict(path_or_state: str | Path | dict) -> dict[str, torch.Tensor]:
    if isinstance(path_or_state, dict):
        return path_or_state
    path = Path(path_or_state)
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("LoRA is .safetensors; install safetensors: pip install safetensors") from None
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _get_attr(obj, name):
    return obj[int(name)] if name.isdigit() else getattr(obj, name)


def _resolve_module(model: nn.Module, base_key: str) -> tuple[nn.Module, str, nn.Module] | None:
    """``(parent, leaf_name, module)`` for a dotted path (numeric parts index containers), else ``None``."""
    *parents, leaf = base_key.split(".")
    try:
        parent = model
        for name in parents:
            parent = _get_attr(parent, name)
        return parent, leaf, _get_attr(parent, leaf)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _extract_layer_index(base_key: str) -> int | None:
    m = _LAYER_INDEX_RE.search(base_key)
    return int(m.group(1)) if m else None


def _stage_bucket(idx: int | None, min_idx: int, max_idx: int) -> int:
    """Depth third of block ``idx`` within ``min_idx..max_idx``: 0 early, 1 mid, 2 late (1 when unknown)."""
    if idx is None or min_idx >= max_idx:
        return 1
    pos = (idx - min_idx) / (max_idx - min_idx)
    if pos <= 1.0 / 3.0:
        return 0
    if pos <= 2.0 / 3.0:
        return 1
    return 2


def layer_in_group(idx: int | None, layer_group: str, min_idx: int, max_idx: int) -> bool:
    """Whether block ``idx`` lies in ``layer_group`` (all / first / middle / last third of ``min_idx..max_idx``).

    The thirds do not overlap. Keys without a block index, and single-depth models, are always kept.
    """
    if layer_group == "all" or idx is None or min_idx >= max_idx:
        return True
    if layer_group not in _LAYER_GROUPS:
        raise ValueError(f"layer_group must be all/first/middle/last, got {layer_group!r}")
    return _stage_bucket(idx, min_idx, max_idx) == _LAYER_GROUPS[layer_group]


@dataclass(slots=True)
class _AdapterTensors:
    down: torch.Tensor | None = None
    up: torch.Tensor | None = None
    alpha: float | None = None
    dora_mag: torch.Tensor | None = None


class MultiLoRALinear(nn.Module):
    """Linear wrapper applying any number of LoRA / DoRA adapters on top of the frozen base layer.

    A LoRA adapter adds ``strength * alpha/rank * x A^T B^T``. A DoRA adapter with a per-output-row magnitude
    ``m`` uses ``W' = m * (W + alpha/rank * B A) / ||W + alpha/rank * B A||_row`` (the formula
    ``inject_trainable_lora`` trains) and adds ``strength * x (W' - W)^T``.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self._adapter_params: nn.ModuleList = nn.ModuleList()  # one nn.ParameterDict per adapter
        self._strengths: list[float] = []
        self._alpha_ratios: list[float] = []
        self._roles: list[str] = []

    def add_adapter(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        *,
        scale: float,
        alpha: float | None = None,
        dora_mag: torch.Tensor | None = None,
        role: str = "style",
    ) -> None:
        rank = int(down.shape[0]) if down.ndim == 2 else 0
        # Adapters load on CPU; match the base layer so forward() has no device/dtype mismatch.
        dev, dtype = self.linear.weight.device, self.linear.weight.dtype
        ad = {
            "down": nn.Parameter(down.to(device=dev, dtype=dtype), requires_grad=False),
            "up": nn.Parameter(up.to(device=dev, dtype=dtype), requires_grad=False),
        }
        out_features = self.linear.out_features
        if dora_mag is not None and dora_mag.numel() == out_features and dora_mag.shape[0] == out_features:
            ad["dora_mag"] = nn.Parameter(dora_mag.reshape(-1).to(device=dev, dtype=dtype), requires_grad=False)
        self._adapter_params.append(nn.ParameterDict(ad))
        self._strengths.append(float(scale))
        self._alpha_ratios.append(float(alpha) / max(1, rank) if alpha is not None else 1.0)
        self._roles.append(_norm_role(role))

    def effective_scales(self) -> list[float]:
        """Per-adapter ``strength * alpha / rank`` (the multiplier on a LoRA delta)."""
        return [s * r for s, r in zip(self._strengths, self._alpha_ratios, strict=True)]

    def set_scale_normalization(
        self,
        *,
        max_total_scale: float = 1.5,
        role_budgets: dict[str, float] | None = None,
    ) -> None:
        """Shrink strengths so each role's total ``|effective scale|`` fits its budget, then the overall total."""
        for role, budget in (role_budgets or {}).items():
            idxs = [i for i, r in enumerate(self._roles) if r == role]
            eff = self.effective_scales()
            total = sum(abs(eff[i]) for i in idxs)
            if budget > 0 and total > budget:
                for i in idxs:
                    self._strengths[i] *= budget / total
        total = sum(abs(s) for s in self.effective_scales())
        if total > float(max_total_scale) and total > 1e-8:
            fac = float(max_total_scale) / total
            self._strengths = [s * fac for s in self._strengths]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        for ad, strength, ratio in zip(self._adapter_params, self._strengths, self._alpha_ratios, strict=True):
            if "dora_mag" in ad:
                w = self.linear.weight
                merged = w + ratio * (ad["up"] @ ad["down"])
                w_dora = (ad["dora_mag"] / merged.norm(dim=1).clamp_min(1e-6)).unsqueeze(1) * merged
                out = out + strength * F.linear(x, w_dora - w)
            else:
                out = out + (strength * ratio) * (x @ ad["down"].T @ ad["up"].T)
        return out


def _extract_adapters(state: dict[str, torch.Tensor], prefix: str = "") -> dict[str, _AdapterTensors]:
    out: dict[str, _AdapterTensors] = {}
    for key, value in state.items():
        key = key.removeprefix(prefix)
        match = next(
            ((key[: -len(sfx)], kind) for kind, sfxs in _SUFFIX_KINDS for sfx in sfxs if key.endswith(sfx)), None
        )
        if match is None:
            continue
        base, kind = match
        adapter = out.setdefault(base, _AdapterTensors())
        if kind == "alpha":
            try:
                value = float(value.item()) if torch.is_tensor(value) else float(value)
            except (RuntimeError, TypeError, ValueError):  # malformed alpha: fall back to alpha == rank
                continue
        setattr(adapter, kind, value)
    return out


def apply_loras(
    model: nn.Module,
    lora_specs: list[tuple[str | Path | dict, float] | tuple[str | Path | dict, float, str]],
    prefix: str = "",
    *,
    normalize_scales: bool = True,
    max_total_scale: float = 1.5,
    role_budgets: dict[str, float] | None = None,
    stage_policy: str = "auto",
    role_stage_weights: dict[str, tuple[float, float, float]] | None = None,
    layer_group: str = "all",
) -> tuple[nn.Module, int]:
    """Wrap every adapted Linear in ``model`` with a ``MultiLoRALinear``; returns ``(model, layers_touched)``.

    ``lora_specs`` items are ``(path_or_state, scale[, role])``. ``stage_policy`` (``auto`` picks
    ``character_focus`` when both character and style adapters are present) and ``role_stage_weights``
    reweight each role by depth third. ``layer_group`` (all/first/middle/last) keeps one depth third: early
    DiT layers mostly shape structure, middle ones detail, late ones aesthetics.
    """
    by_layer: dict[str, list[tuple[_AdapterTensors, float, str]]] = {}
    for spec in lora_specs:
        path_or_state, scale = spec[0], float(spec[1])
        role = _norm_role(spec[2] if len(spec) >= 3 else None)
        for base, ad in _extract_adapters(_get_lora_state_dict(path_or_state), prefix=prefix).items():
            if ad.down is not None and ad.up is not None:
                by_layer.setdefault(base, []).append((ad, scale, role))

    policy = str(stage_policy or "off").strip().lower()
    if policy == "auto":
        roles_present = {r for ads in by_layer.values() for _, _, r in ads}
        policy = "character_focus" if {"character", "style"} <= roles_present else "off"
    stage_weights = dict(_STAGE_POLICIES.get(policy, {}))
    for role, weights in (role_stage_weights or {}).items():
        try:
            if len(weights) >= 3:
                stage_weights[str(role).strip().lower()] = (float(weights[0]), float(weights[1]), float(weights[2]))
        except (TypeError, ValueError):
            continue

    known_indices = [i for i in map(_extract_layer_index, by_layer) if i is not None]
    min_idx, max_idx = (min(known_indices), max(known_indices)) if known_indices else (0, 0)
    lg = str(layer_group or "all").strip().lower()
    budgets = {str(k).strip().lower(): float(v) for k, v in (role_budgets or {}).items()}

    touched = 0
    for base_key, adapters in by_layer.items():
        layer_idx = _extract_layer_index(base_key)
        if not layer_in_group(layer_idx, lg, min_idx, max_idx):
            continue
        resolved = _resolve_module(model, base_key)
        if resolved is None:
            continue
        parent, leaf, mod = resolved
        if isinstance(mod, nn.Linear):
            wrapper = MultiLoRALinear(mod)
        elif isinstance(mod, MultiLoRALinear):
            wrapper = mod
        else:
            continue

        stage_i = _stage_bucket(layer_idx, min_idx, max_idx)
        for ad, scale, role in adapters:
            stage_w = stage_weights.get(role, stage_weights.get("other", (1.0, 1.0, 1.0)))[stage_i]
            wrapper.add_adapter(
                ad.down, ad.up, scale=scale * float(stage_w), alpha=ad.alpha, dora_mag=ad.dora_mag, role=role
            )
        if normalize_scales:
            wrapper.set_scale_normalization(max_total_scale=float(max_total_scale), role_budgets=budgets)
        if leaf.isdigit():
            parent[int(leaf)] = wrapper
        else:
            setattr(parent, leaf, wrapper)
        touched += 1

    return model, touched


# Default injection targets: attention projections and MLP layers, matched by
# substring against each Linear's dotted module path. Covers the DiT blocks
# without touching embedders/heads.
DEFAULT_TARGETS: tuple[str, ...] = ("qkv", "q_proj", "k_proj", "v_proj", "proj", "fc1", "fc2", "mlp")


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

    def adapter_state(self, base_key: str) -> dict[str, torch.Tensor]:
        """Adapter tensors keyed for the inference loader (``apply_loras``)."""
        out = {
            f"{base_key}.lora_down.weight": self.lora_down.detach().cpu(),
            f"{base_key}.lora_up.weight": self.lora_up.detach().cpu(),
            f"{base_key}.alpha": torch.tensor(float(self.alpha)),
        }
        if self.dora_magnitude is not None:
            out[f"{base_key}.dora_magnitude_vector"] = self.dora_magnitude.detach().cpu()
        return out


def _iter_target_linears(model: nn.Module, targets: Sequence[str]) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    """``(parent_module, attr_name, dotted_path, linear)`` for every Linear whose path contains a target."""
    found: list[tuple[nn.Module, str, str, nn.Linear]] = []
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


def _filter_layer_group(
    found: list[tuple[nn.Module, str, str, nn.Linear]], layer_group: str
) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    """Keep one depth third of the blocks, with the same split ``models.lora.apply_loras`` uses at inference."""
    known = [i for i in (_extract_layer_index(path) for _, _, path, _ in found) if i is not None]
    lo, hi = (min(known), max(known)) if known else (0, 0)
    return [f for f in found if layer_in_group(_extract_layer_index(f[2]), layer_group, lo, hi)]


def inject_trainable_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    use_dora: bool = False,
    targets: Sequence[str] | None = None,
    layer_group: str = "all",
) -> tuple[list[nn.Parameter], int]:
    """Freeze ``model`` and wrap target Linear layers with trainable adapters.

    ``layer_group`` (all/first/middle/last) limits adapters to a third of block depth.
    Returns ``(trainable_params, num_layers_wrapped)``.
    """
    targets = tuple(targets) if targets is not None else DEFAULT_TARGETS

    # Freeze everything first; adapters below re-enable grad on their own params.
    for p in model.parameters():
        p.requires_grad = False

    found = _filter_layer_group(_iter_target_linears(model, targets), str(layer_group or "all").strip().lower())
    for parent, attr, _path, linear in found:
        setattr(parent, attr, TrainableLoRALinear(linear, rank=rank, alpha=alpha, use_dora=use_dora))
    return [p for p in model.parameters() if p.requires_grad], len(found)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect adapter-only tensors from all injected layers, keyed for inference."""
    out: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        if isinstance(module, TrainableLoRALinear):
            # ``path`` ends at the wrapper; the inference loader resolves the same
            # dotted path to this module and reads its adapter tensors.
            out.update(module.adapter_state(path))
    return out


def unfreeze_control_encoder(model: nn.Module) -> int:
    """Re-enable gradients on the in-DiT ControlNet encoder (for LoRA + control training)."""
    raw = model.module if hasattr(model, "module") else model
    enc = getattr(raw, "control_encoder", None)
    if enc is None:
        return 0
    n = 0
    for p in enc.parameters():
        p.requires_grad = True
        n += p.numel()
    return n
