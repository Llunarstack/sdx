"""
Generic ``torch.nn`` inspection: per-child parameter counts and a compact module tree.

Useful for DiT, ViT, and any large ``nn.Module``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn


def child_parameter_summary(module: nn.Module, *, top_k: int = 24) -> List[Dict[str, Any]]:
    """
    Direct-child parameter counts (sorted descending by numel).

    Returns list of dicts: ``name``, ``type``, ``parameters``, ``trainable_parameters``.
    """
    rows: List[Tuple[str, str, int, int]] = []
    for name, child in module.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n == 0:
            continue
        nt = sum(p.numel() for p in child.parameters() if p.requires_grad)
        rows.append((name, type(child).__name__, n, nt))
    rows.sort(key=lambda x: x[2], reverse=True)
    out: List[Dict[str, Any]] = []
    for name, typ, n, nt in rows[:top_k]:
        out.append(
            {
                "name": name,
                "type": typ,
                "parameters": n,
                "trainable_parameters": nt,
            }
        )
    return out


def format_module_tree(
    module: nn.Module,
    *,
    max_depth: int = 4,
    max_children: int = 12,
    _depth: int = 0,
    _indent: str = "",
) -> List[str]:
    """Compact ASCII lines: child name, type, subtree param count."""
    lines: List[str] = []
    total = sum(p.numel() for p in module.parameters())
    if _depth == 0:
        lines.append(f"{type(module).__name__}  total_params={total:,}")

    if _depth >= max_depth:
        return lines

    all_children = list(module.named_children())
    children = all_children[:max_children]
    for name, child in children:
        cn = sum(p.numel() for p in child.parameters())
        lines.append(f"{_indent}{name}  {type(child).__name__}  ({cn:,})")
        lines.extend(
            format_module_tree(
                child,
                max_depth=max_depth,
                max_children=max_children,
                _depth=_depth + 1,
                _indent=_indent + "  ",
            )
        )

    if _depth == 0 and len(all_children) > max_children:
        lines.append(f"{_indent}... ({len(all_children) - max_children} more direct children omitted)")

    return lines
