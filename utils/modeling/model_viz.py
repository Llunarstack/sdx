"""Model inspection: parameter summary by module type (``train.py``) and a compact module tree (``vit_inspect``)."""

from __future__ import annotations

import torch.nn as nn


def print_model_summary(model: nn.Module) -> None:
    """Total / trainable parameters, fp32 size and the parameter share per leaf-module type."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_type: dict[str, int] = {}
    n_layers = 0
    for module in model.modules():
        if next(module.children(), None) is not None:
            continue
        n = sum(p.numel() for p in module.parameters())
        by_type[type(module).__name__] = by_type.get(type(module).__name__, 0) + n
        n_layers += n > 0

    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size: {total * 4 / (1024 * 1024):.2f} MB")
    print(f"Number of Layers: {n_layers}")
    print("\nParameter Distribution by Module Type:")
    print("-" * 50)
    for module_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"{module_type:20s}: {count:>12,} ({100.0 * count / max(total, 1):5.1f}%)")
    print("=" * 80)


def format_module_tree(
    module: nn.Module,
    *,
    max_depth: int = 4,
    max_children: int = 12,
    _depth: int = 0,
    _indent: str = "",
) -> list[str]:
    """Compact ASCII lines: child name, type, subtree param count."""
    lines: list[str] = []
    if _depth == 0:
        lines.append(f"{type(module).__name__}  total_params={sum(p.numel() for p in module.parameters()):,}")
    if _depth >= max_depth:
        return lines

    all_children = list(module.named_children())
    for name, child in all_children[:max_children]:
        lines.append(f"{_indent}{name}  {type(child).__name__}  ({sum(p.numel() for p in child.parameters()):,})")
        lines.extend(
            format_module_tree(
                child, max_depth=max_depth, max_children=max_children, _depth=_depth + 1, _indent=_indent + "  "
            )
        )
    if _depth == 0 and len(all_children) > max_children:
        lines.append(f"{_indent}... ({len(all_children) - max_children} more direct children omitted)")
    return lines
