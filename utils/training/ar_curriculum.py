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
        x = int(v)
    except Exception:
        return 0
    if x <= 0:
        return 0
    if x <= 2:
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
    for p in str(spec).split(","):
        k = p.strip().lower()
        if not k or k not in _VALID_ORDERS or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def choose_ar_order_for_step(step: int, *, base_order: str, mix_orders: Sequence[str]) -> str:
    """Deterministic step-indexed AR traversal order."""
    if not mix_orders:
        k = str(base_order or "raster").strip().lower()
        return k if k in _VALID_ORDERS else "raster"
    idx = max(0, int(step)) % len(mix_orders)
    return str(mix_orders[idx])


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
    rs, re = int(ramp_start), int(ramp_end)
    if re <= rs:
        return int(end_blocks)
    if step <= rs:
        return int(start_blocks)
    if step >= re:
        return int(end_blocks)
    t = (float(step) - float(rs)) / float(re - rs)
    # Snap to valid discrete values {0,2,4} when close
    v = (1.0 - t) * float(start_blocks) + t * float(end_blocks)
    if v < 1.0:
        return 0
    if v < 3.0:
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
    b_base = normalize_ar_blocks(base_blocks)
    b_start = normalize_ar_blocks(curriculum_start_blocks if int(curriculum_start_blocks) >= 0 else 0)
    b_target = normalize_ar_blocks(curriculum_target_blocks if int(curriculum_target_blocks) >= 0 else b_base)
    s = max(0, int(step))
    if mode == "step":
        b = normalize_ar_blocks(
            num_ar_blocks_for_step(
                s,
                warmup_steps=max(0, int(warmup_steps)),
                target_blocks=b_target,
                start_blocks=b_start,
            )
        )
    elif mode == "linear":
        rs = max(0, int(ramp_start))
        re = max(rs + 1, int(ramp_end)) if int(ramp_end) > 0 else max(rs + 1, rs + max(1, int(warmup_steps)))
        b = normalize_ar_blocks(
            num_ar_blocks_linear_ramp(
                s,
                ramp_start=rs,
                ramp_end=re,
                start_blocks=b_start,
                end_blocks=b_target,
            )
        )
    else:
        b = b_base

    if b <= 0:
        return 0, "raster"
    mix = parse_ar_order_mix(order_mix)
    order = choose_ar_order_for_step(s, base_order=base_order, mix_orders=mix)
    return b, order
