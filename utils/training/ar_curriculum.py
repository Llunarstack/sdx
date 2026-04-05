"""
Training-time schedules for block-wise AR strength (``num_ar_blocks``).

Use from ``train.py`` when you want to start full bidirectional and ramp AR mid-training.
"""

from __future__ import annotations


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
