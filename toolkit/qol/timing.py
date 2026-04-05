"""Lightweight timing for training loops (stdlib only)."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class StepTimer:
    """Track rolling average step time and optional ETA."""

    def __init__(self, ema_decay: float = 0.98) -> None:
        self.ema_decay = float(ema_decay)
        self._last_start: Optional[float] = None
        self.avg_sec: float = 0.0
        self.steps: int = 0

    def tick_start(self) -> None:
        self._last_start = time.perf_counter()

    def tick_end(self) -> float:
        if self._last_start is None:
            return 0.0
        dt = time.perf_counter() - self._last_start
        self.steps += 1
        if self.avg_sec <= 0:
            self.avg_sec = dt
        else:
            d = self.ema_decay
            self.avg_sec = d * self.avg_sec + (1.0 - d) * dt
        return dt

    def steps_per_sec(self) -> float:
        return 1.0 / self.avg_sec if self.avg_sec > 0 else 0.0

    def eta_str(self, done: int, total: int) -> str:
        if total <= 0 or self.avg_sec <= 0:
            return "?"
        remain = max(0, total - done)
        sec = remain * self.avg_sec
        if sec < 60:
            return f"{sec:.0f}s"
        if sec < 3600:
            return f"{sec / 60:.1f}m"
        return f"{sec / 3600:.1f}h"


def timed(name: str = "", log_fn: Optional[Callable[[str], None]] = None) -> Callable[[F], F]:
    """
    Decorator: log elapsed seconds. ``log_fn`` defaults to ``print``.
    """

    def deco(fn: F) -> F:
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                msg = f"[timed] {label}: {dt:.4f}s"
                (log_fn or print)(msg)

        return wrapped  # type: ignore[return-value]

    return deco


__all__ = ["StepTimer", "timed"]
