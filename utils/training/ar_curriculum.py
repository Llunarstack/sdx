"""
Training-time schedules for block-wise AR strength (``num_ar_blocks``).

Use from ``train.py`` when you want to start full bidirectional and ramp AR mid-training.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

_VALID_BLOCKS = (0, 2, 4)
_VALID_ORDERS = ("raster", "zorder", "snake", "spiral")


def normalize_ar_blocks(v: int) -> int:
    """Snap arbitrary int to supported AR regime {0,2,4}."""
    try:
        blocks = int(v)
    except Exception:
        return 0
    if blocks <= 0:
        return 0
    if blocks <= 2:
        return 2
    return 4


def parse_ar_order_mix(spec: str | None) -> List[str]:
    """
    Parse comma-separated AR orders (e.g. ``"raster,zorder,snake"``).

    Invalid entries are ignored; duplicates are removed while preserving order.
    """
    if not spec or not str(spec).strip():
        return []
    out: List[str] = []
    seen = set()
    for raw_order in str(spec).split(","):
        order = raw_order.strip().lower()
        if not order or order not in _VALID_ORDERS or order in seen:
            continue
        seen.add(order)
        out.append(order)
    return out


def choose_ar_order_for_step(step: int, *, base_order: str, mix_orders: Sequence[str]) -> str:
    """Deterministic step-indexed AR traversal order."""
    if not mix_orders:
        fallback_order = str(base_order or "raster").strip().lower()
        return fallback_order if fallback_order in _VALID_ORDERS else "raster"
    order_index = max(0, int(step)) % len(mix_orders)
    return str(mix_orders[order_index])


def num_ar_blocks_for_step(
    step: int,
    *,
    warmup_steps: int,
    target_blocks: int,
    start_blocks: int = 0,
) -> int:
    """
    Piecewise constant curriculum: ``start_blocks`` until ``warmup_steps``, then ``target_blocks``.

    Typical: ``start_blocks=0``, ``target_blocks=2``, ``warmup_steps=10_000``.
    """
    if step < int(warmup_steps):
        return int(start_blocks)
    return int(target_blocks)


def num_ar_blocks_linear_ramp(
    step: int,
    *,
    ramp_start: int,
    ramp_end: int,
    start_blocks: int = 0,
    end_blocks: int = 2,
) -> int:
    """Linearly interpolate between start_blocks and end_blocks over [ramp_start, ramp_end]."""
    ramp_start_step, ramp_end_step = int(ramp_start), int(ramp_end)
    if ramp_end_step <= ramp_start_step:
        return int(end_blocks)
    if step <= ramp_start_step:
        return int(start_blocks)
    if step >= ramp_end_step:
        return int(end_blocks)
    ramp_fraction = (float(step) - float(ramp_start_step)) / float(ramp_end_step - ramp_start_step)
    # Snap to valid discrete values {0,2,4} when close
    interpolated_blocks = (1.0 - ramp_fraction) * float(start_blocks) + ramp_fraction * float(end_blocks)
    if interpolated_blocks < 1.0:
        return 0
    if interpolated_blocks < 3.0:
        return 2
    return 4


def resolve_ar_for_step(
    step: int,
    *,
    base_blocks: int,
    base_order: str,
    curriculum_mode: str = "none",
    warmup_steps: int = 0,
    ramp_start: int = 0,
    ramp_end: int = 0,
    curriculum_start_blocks: int = -1,
    curriculum_target_blocks: int = -1,
    order_mix: str | None = None,
) -> Tuple[int, str]:
    """
    Resolve runtime ``(num_ar_blocks, ar_block_order)`` for a training step.

    curriculum_mode:
      - ``none``: fixed blocks (from base)
      - ``step``: piecewise constant at warmup boundary
      - ``linear``: linear ramp over [ramp_start, ramp_end]
    """
    mode = str(curriculum_mode or "none").strip().lower()
    base_blocks_normalized = normalize_ar_blocks(base_blocks)
    start_blocks_normalized = normalize_ar_blocks(curriculum_start_blocks if int(curriculum_start_blocks) >= 0 else 0)
    target_blocks_normalized = normalize_ar_blocks(
        curriculum_target_blocks if int(curriculum_target_blocks) >= 0 else base_blocks_normalized
    )
    step_index = max(0, int(step))
    if mode == "step":
        resolved_blocks = normalize_ar_blocks(
            num_ar_blocks_for_step(
                step_index,
                warmup_steps=max(0, int(warmup_steps)),
                target_blocks=target_blocks_normalized,
                start_blocks=start_blocks_normalized,
            )
        )
    elif mode == "linear":
        ramp_start_step = max(0, int(ramp_start))
        ramp_end_step = (
            max(ramp_start_step + 1, int(ramp_end))
            if int(ramp_end) > 0
            else max(ramp_start_step + 1, ramp_start_step + max(1, int(warmup_steps)))
        )
        resolved_blocks = normalize_ar_blocks(
            num_ar_blocks_linear_ramp(
                step_index,
                ramp_start=ramp_start_step,
                ramp_end=ramp_end_step,
                start_blocks=start_blocks_normalized,
                end_blocks=target_blocks_normalized,
            )
        )
    else:
        resolved_blocks = base_blocks_normalized

    if resolved_blocks <= 0:
        return 0, "raster"
    order_mix_list = parse_ar_order_mix(order_mix)
    order = choose_ar_order_for_step(step_index, base_order=base_order, mix_orders=order_mix_list)
    return resolved_blocks, order
