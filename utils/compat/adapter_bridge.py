"""
Cross-architecture adapter **bridge** — apply foreign LoRA-family adapters to
the sdx DiT.

A foreign adapter (kohya SD/SDXL, diffusers PEFT, Flux, LyCORIS) names layers
of a *different* network, so exact key translation is impossible. The bridge
instead maps each foreign layer by two architecture-independent coordinates:

- **role** — what the layer does (attn q/k/v/out, mlp in/out); and
- **depth fraction** — how far into the network it sits (0=input, 1=output).

Each foreign ΔW (reconstructed via ``lycoris_math``) is assigned to the sdx
Linear with the same role at the nearest depth, shape-projected, re-factorized
by SVD, and attached through the existing :class:`models.lora.MultiLoRALinear`
stack — so scale normalization, roles, and stage policies all keep working.

Separate q/k/v deltas fold into sdx's fused ``qkv`` as row-block slices.
Text-encoder tensors are skipped here (see ``embedding_bridge``). Transfer is
approximate: style/detail adapters bridge well, exact-character adapters less
so — the coverage report says how much landed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from utils.compat.fingerprint_basis import FingerprintBasis

from utils.compat.asset_sniffer import load_state_dict
from utils.compat.lycoris_math import (
    delta_from_loha,
    delta_from_lokr,
    delta_from_lora,
    project_delta,
    svd_factorize,
)

__all__ = [
    "BridgeReport",
    "bridge_apply",
    "collect_foreign_deltas",
    "map_sdx_targets",
]

# Role detection on foreign key names (kohya underscores and diffusers dots).
# Covers the layer-naming conventions of every popular CivitAI ecosystem:
# SD1.5/SD2/SDXL UNets (to_q/to_out/ff_net), DiT fused qkv, Hunyuan-DiT Wqkv,
# Qwen-Image to_qkv/add_qkv_proj/to_add_out, Lumina/HiDream SwiGLU w1/w2/w3.
_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(\W|_)[Ww]?qkv(\W|_|$)", "qkv"),
    (r"(to_q|q_proj|attn_q)(\W|_|$)", "q"),
    (r"(to_k|k_proj|attn_k)(\W|_|$)", "k"),
    (r"(to_v|v_proj|attn_v)(\W|_|$)", "v"),
    (r"(to_out|to_add_out|out_proj|proj_out|attn(\W|_)proj)", "attn_out"),
    (r"(ff(\W|_)net(\W|_)0|fc1|mlp(\W|_)(0|fc1)|proj(\W|_)mlp|linear_1|w[13](\W|_|$))", "mlp_in"),
    (r"(ff(\W|_)net(\W|_)2|fc2|mlp(\W|_)(2|fc2)|linear_2|w2(\W|_|$))", "mlp_out"),
)

_BLOCK_INDEX_RE = re.compile(
    r"(?:input_blocks|output_blocks|down_blocks|up_blocks|double_blocks|single_blocks|"
    r"joint_blocks|transformer_blocks|blocks|layers)[._](\d+)"
)

# Which foreign roles may land on which sdx module name patterns.
_SDX_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "qkv": (r"\.qkv$",),
    "q": (r"\.q_proj$", r"\.qkv$"),
    "k": (r"\.k_proj$", r"\.qkv$"),
    "v": (r"\.v_proj$", r"\.qkv$"),
    "attn_out": (r"attn.*\.(proj|out_proj|o_proj)$",),
    "mlp_in": (r"(mlp|ff)\w*\.(fc1|0)$", r"\.fc1$"),
    "mlp_out": (r"(mlp|ff)\w*\.(fc2|2)$", r"\.fc2$"),
}


@dataclass(slots=True)
class ForeignDelta:
    key: str
    role: str
    depth_fraction: float
    delta: torch.Tensor


@dataclass(slots=True)
class BridgeReport:
    """What happened when a foreign adapter was bridged onto sdx."""

    foreign_layers: int = 0
    mapped: int = 0
    skipped_text_encoder: int = 0
    skipped_role: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.mapped / self.foreign_layers if self.foreign_layers else 0.0


def _detect_role(key: str) -> str:
    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, key):
            return role
    return "other"


def _depth_fraction(key: str, max_index: int) -> float:
    m = _BLOCK_INDEX_RE.search(key)
    if not m:
        return 0.5 if "middle_block" in key or "mid_block" in key else 0.5
    return int(m.group(1)) / max(1, max_index)


def _group_adapter_tensors(state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state.items():
        if "." not in k:
            continue
        base, _, part = k.rpartition(".")
        # kohya nests one more level: base.lora_down.weight
        if part == "weight" and base.rpartition(".")[2] in ("lora_down", "lora_up", "lora_A", "lora_B"):
            base, _, sub = base.rpartition(".")
            part = f"{sub}.weight"
        grouped.setdefault(base, {})[part] = v
    return grouped


def _delta_from_parts(parts: dict[str, torch.Tensor]) -> torch.Tensor | None:
    alpha = None
    if "alpha" in parts:
        try:
            alpha = float(parts["alpha"].item())
        except Exception:
            alpha = None
    if "hada_w1_a" in parts and "hada_w2_b" in parts:
        return delta_from_loha(
            parts["hada_w1_a"], parts["hada_w1_b"], parts["hada_w2_a"], parts["hada_w2_b"], alpha=alpha
        )
    if "lokr_w1" in parts or "lokr_w1_a" in parts:
        return delta_from_lokr(
            parts.get("lokr_w1"),
            parts.get("lokr_w2"),
            w1_a=parts.get("lokr_w1_a"),
            w1_b=parts.get("lokr_w1_b"),
            w2_a=parts.get("lokr_w2_a"),
            w2_b=parts.get("lokr_w2_b"),
            alpha=alpha,
        )
    down = parts.get("lora_down.weight", parts.get("lora_A.weight"))
    up = parts.get("lora_up.weight", parts.get("lora_B.weight"))
    if down is not None and up is not None:
        return delta_from_lora(down, up, alpha=alpha)
    if "diff" in parts:
        d = parts["diff"]
        return d.reshape(d.shape[0], -1).float() if d.ndim >= 2 else None
    return None


def _split_fused_single_block(base: str, delta: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
    """
    Split Flux/HiDream ``single_blocks`` fused matrices into bridgeable roles.

    ``linear1`` stacks qkv + mlp-in rows: (3d + mlp_hidden, d) → qkv rows are
    the first 3d. ``linear2`` concatenates attn-out + mlp-out columns:
    (d, d + mlp_hidden) → attn columns are the first d.
    """
    if "single" not in base or "blocks" not in base:
        return None
    rows, cols = int(delta.shape[0]), int(delta.shape[1])
    if re.search(r"linear[._]?1$", base) and rows > 3 * cols:
        return [("qkv", delta[: 3 * cols]), ("mlp_in", delta[3 * cols :])]
    if re.search(r"(linear[._]?2|proj[._]?out)$", base) and cols > 3 * rows:
        return [("attn_out", delta[:, :rows]), ("mlp_out", delta[:, rows:])]
    return None


def collect_foreign_deltas(
    path_or_state: str | Path | dict,
    *,
    report: BridgeReport | None = None,
) -> list[ForeignDelta]:
    """Reconstruct role/depth-tagged ΔW matrices from a foreign adapter file."""
    state = load_state_dict(path_or_state)
    rep = report if report is not None else BridgeReport()
    grouped = _group_adapter_tensors(state)
    max_idx = 0
    for base in grouped:
        m = _BLOCK_INDEX_RE.search(base)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    out: list[ForeignDelta] = []
    for base, parts in grouped.items():
        if base.startswith(("lora_te", "text_encoder", "te1", "te2")) or ".text_model." in base:
            rep.skipped_text_encoder += 1
            continue
        delta = _delta_from_parts(parts)
        if delta is None:
            continue
        rep.foreign_layers += 1
        frac = _depth_fraction(base, max_idx)
        split = _split_fused_single_block(base, delta)
        if split is not None:
            out.extend(ForeignDelta(base, r, frac, d) for r, d in split)
            continue
        role = _detect_role(base)
        if role == "other":
            rep.skipped_role += 1
            continue
        out.append(ForeignDelta(base, role, frac, delta))
    return out


def map_sdx_targets(model: nn.Module) -> dict[str, list[tuple[str, float, nn.Linear]]]:
    """Enumerate bridgeable sdx Linears as ``role → [(name, depth_fraction, module)]``."""
    linears: list[tuple[str, nn.Linear]] = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    max_idx = 0
    for name, _ in linears:
        m = _BLOCK_INDEX_RE.search(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    targets: dict[str, list[tuple[str, float, nn.Linear]]] = {}
    for role, patterns in _SDX_ROLE_PATTERNS.items():
        for name, mod in linears:
            if any(re.search(p, name) for p in patterns):
                targets.setdefault(role, []).append((name, _depth_fraction(name, max_idx), mod))
    return targets


def _fused_qkv_slice(role: str, out_features: int) -> tuple[int, int] | None:
    third = out_features // 3
    return {"q": (0, third), "k": (third, 2 * third), "v": (2 * third, 3 * third)}.get(role)


def bridge_apply(
    model: nn.Module,
    adapter_path_or_state: str | Path | dict,
    *,
    scale: float = 1.0,
    rank: int = 32,
    role: str = "style",
    fingerprint: FingerprintBasis | None = None,
    strip_source_fingerprint: float = 0.0,
) -> BridgeReport:
    """
    Bridge a foreign adapter onto ``model`` in place.

    Returns a :class:`BridgeReport`; check ``coverage`` — below ~0.5 the
    adapter targets structure sdx doesn't share and the effect will be weak.

    ``strip_source_fingerprint`` (0..1) removes the source base-model fingerprint
    from each bridged ΔW using ``fingerprint`` (a :class:`FingerprintBasis` built
    for the adapter's source family via ``fingerprint_basis.build_fingerprint_basis``),
    so foreign adapters don't drag their base model's "AI look" into outputs.
    Higher = cleaner but erodes more of the concept; tune with the critic loop.
    """
    strip = float(max(0.0, min(1.0, strip_source_fingerprint)))
    from models.lora import MultiLoRALinear, _resolve_module, _set_attr

    report = BridgeReport()
    deltas = collect_foreign_deltas(adapter_path_or_state, report=report)
    targets = map_sdx_targets(model)
    for fd in deltas:
        candidates = targets.get(fd.role) or []
        if not candidates:
            report.skipped_role += 1
            continue
        name, _, module = min(candidates, key=lambda c: abs(c[1] - fd.depth_fraction))
        base = module.linear if isinstance(module, MultiLoRALinear) else module
        out_f, in_f = int(base.out_features), int(base.in_features)
        qkv_slice = None
        if fd.role in ("q", "k", "v") and out_f == 3 * in_f:
            qkv_slice = _fused_qkv_slice(fd.role, out_f)
        proj_out = (qkv_slice[1] - qkv_slice[0]) if qkv_slice else out_f
        delta = project_delta(fd.delta, (proj_out, in_f), rank=rank)
        if strip > 0.0 and fingerprint is not None:
            from utils.compat.fingerprint_basis import project_out

            delta = project_out(delta, fingerprint, fd.role, fd.depth_fraction, strip)
        if qkv_slice is not None:
            full = delta.new_zeros(out_f, in_f)
            full[qkv_slice[0] : qkv_slice[1], :] = delta
            delta = full
        down, up = svd_factorize(delta, rank)
        parent, leaf, mod = _resolve_module(model, name)
        if parent is None or leaf is None or mod is None:
            continue
        if isinstance(mod, MultiLoRALinear):
            wrapper = mod
        else:
            wrapper = MultiLoRALinear(mod)
            _set_attr(parent, leaf, wrapper)
            for r, entries in targets.items():  # keep the registry pointing at live modules
                targets[r] = [(n, f, wrapper if n == name else m) for n, f, m in entries]
        wrapper.add_adapter(down, up, scale=float(scale), alpha=None, role=role)
        report.mapped += 1
    report.notes.append(f"coverage {report.coverage:.0%} of {report.foreign_layers} foreign layers")
    return report
