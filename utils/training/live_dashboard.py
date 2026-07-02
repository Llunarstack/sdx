"""Real-time training dashboard for the terminal.

Renders a live panel (progress, loss trend, throughput, ETA, GPU) that refreshes
in place while training runs. Degrades to plain periodic prints if ``rich`` is
unavailable. Rank-0 only; call :meth:`update` from the training log step.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-12:
        return _SPARK[0] * len(vals)
    return "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1)))] for v in vals)


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # <=0 or NaN
        return "--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class LiveDashboard:
    """Accumulates training metrics and renders them live."""

    def __init__(self, total_steps: int, *, model_name: str = "", history: int = 60, enabled: bool = True):
        self.total_steps = int(total_steps) if total_steps and total_steps > 0 else 0
        self.model_name = model_name
        self.start_time = time.time()
        self.loss_hist: Deque[float] = deque(maxlen=history)
        self.best_loss = float("inf")
        self.ema_loss: Optional[float] = None
        self.enabled = enabled

        self._live = None
        self._console = None
        if not enabled:
            return
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            self._live = Live(self._render(0, 0.0, 0.0, 0.0, {}), console=self._console, refresh_per_second=8)
            self._live.start()
        except Exception:
            # rich missing or non-tty: fall back to plain prints in update().
            self._live = None

    def _trend(self) -> str:
        if len(self.loss_hist) < 6:
            return "warming up"
        recent = list(self.loss_hist)
        first = sum(recent[: len(recent) // 3]) / max(1, len(recent) // 3)
        last = sum(recent[-len(recent) // 3 :]) / max(1, len(recent) // 3)
        if last < first * 0.995:
            return "improving ↓"
        if last > first * 1.02:
            return "rising ↑ (watch LR / divergence)"
        return "flat →"

    def _health(self) -> str:
        if len(self.loss_hist) < 4:
            return "starting"
        trend = self._trend()
        if "rising" in trend:
            return "unstable — check LR / batch"
        if "improving" in trend:
            return "healthy"
        return "plateau — normal late-training"

    def _render(self, step: int, loss: float, steps_per_sec: float, lr: float, extra: dict):
        try:
            from rich.panel import Panel
            from rich.table import Table
        except Exception:
            return ""

        elapsed = time.time() - self.start_time
        frac = (step / self.total_steps) if self.total_steps else 0.0
        eta = (self.total_steps - step) / steps_per_sec if (self.total_steps and steps_per_sec > 0) else 0.0
        bar_w = 34
        filled = int(frac * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)

        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(justify="right", style="cyan", no_wrap=True)
        table.add_column(style="white")

        pct = f"{frac * 100:5.1f}%" if self.total_steps else "  n/a"
        step_str = f"{step:,}/{self.total_steps:,}" if self.total_steps else f"{step:,}"
        table.add_row("progress", f"{bar} {pct}   step {step_str}")
        table.add_row("loss", f"{loss:.4f}   best {self.best_loss:.4f}   ema {self.ema_loss or loss:.4f}   [{self._trend()}]")
        table.add_row("health", self._health())
        table.add_row("trend", _sparkline(list(self.loss_hist)))
        gpu = extra.get("gpu_mem", "")
        table.add_row("throughput", f"{steps_per_sec:.2f} steps/s   {extra.get('img_per_sec', 0.0):.1f} img/s   lr {lr:.2e}")
        table.add_row("time", f"elapsed {_fmt_eta(elapsed)}   eta {_fmt_eta(eta)}" + (f"   gpu {gpu}" if gpu else ""))
        aux = extra.get("aux", "")
        if aux:
            table.add_row("aux", aux)
        title = f"SDX training · {self.model_name}" if self.model_name else "SDX training"
        return Panel(table, title=title, border_style="green")

    def update(self, step: int, loss: float, steps_per_sec: float, lr: float, **extra) -> None:
        if not self.enabled:
            return
        if loss == loss:  # not NaN
            self.loss_hist.append(loss)
            self.best_loss = min(self.best_loss, loss)
            self.ema_loss = loss if self.ema_loss is None else (0.9 * self.ema_loss + 0.1 * loss)

        gpu = ""
        if torch is not None and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu = f"{used:.1f}/{total:.0f}GB"
        extra.setdefault("gpu_mem", gpu)

        if self._live is not None:
            self._live.update(self._render(step, loss, steps_per_sec, lr, extra))
        else:
            eta = (self.total_steps - step) / steps_per_sec if (self.total_steps and steps_per_sec > 0) else 0.0
            print(
                f"[dash] step {step}/{self.total_steps or '?'} loss={loss:.4f} best={self.best_loss:.4f} "
                f"{steps_per_sec:.2f}it/s eta={_fmt_eta(eta)} {self._trend()} {gpu}",
                flush=True,
            )

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
