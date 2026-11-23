"""Opt-in sampling and finishing techniques ("ideas"): each one is a small function at one hook point.

    sample.py ... --ideas noise_pyramid,delta_channel_center=0.5,img_blue_noise_dither
    sample.py --list-ideas
    python -m scripts.tools bench ablate --ckpt best.pt --ideas-each      # each idea vs the baseline

Strength 1 is the suggested amount and 0 is an exact no-op; every idea is off unless named. Hooks, in sampling order:

========  =======================================  =================================================
hook      called with                              where
========  =======================================  =================================================
noise     ``x`` (B, C, H, W) initial noise         once, before the first step (not on img2img starts)
scale     ``w`` CFG scale (float)                  every step
delta     ``delta, d_cond, w`` (data space)        every CFG step: ``guided = d_cond + (w - 1) * delta``
x0        ``x0`` predicted clean latent            every step, after thresholding
step      ``x_new, x_prev, x0``                    every step, after the solver update
image     ``rgb`` float32 HWC in [0, 1]            once per saved image, after every other finish
========  =======================================  =================================================
"""

from __future__ import annotations

import difflib
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HOOKS = ("noise", "scale", "delta", "x0", "step", "image")


@dataclass(frozen=True)
class Idea:
    name: str
    hook: str
    fn: Callable[..., Any]
    summary: str
    quantizes: bool = False  # image ideas whose output is already on the 8-bit grid (dithers)


IDEAS: dict[str, Idea] = {}


def idea(hook: str, summary: str, *, quantizes: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` (named after the function) at ``hook``; ``fn(value, *extra, ctx)`` returns the new value."""
    if hook not in HOOKS:
        raise ValueError(f"unknown hook {hook!r}; expected one of {HOOKS}")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if fn.__name__ in IDEAS:
            raise ValueError(f"duplicate idea {fn.__name__!r}")
        IDEAS[fn.__name__] = Idea(fn.__name__, hook, fn, summary, quantizes)
        return fn

    return deco


@dataclass
class Ctx:
    """Per-idea state for one sampling run: strength, step position, seed and scratch ``state``."""

    name: str
    strength: float
    seed: int = 0
    native: int = 0  # the DiT's native latent side (0 = unknown)
    prompt: str = ""
    i: int = 0
    n: int = 1
    shape: tuple[int, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def p(self) -> float:
        """Sampling progress: 0 at the first (noisiest) step, 1 at the last."""
        return 0.0 if self.n <= 1 else min(1.0, self.i / (self.n - 1))

    def _key(self, salt: int = 0) -> int:
        k = int(self.seed) * 1_000_003 + zlib.crc32(self.name.encode()) + 7919 * int(self.i) + 104_729 * int(salt)
        return k % 2**63

    def randn(self, like: Any, salt: int = 0) -> Any:
        """Seeded Gaussian noise shaped like the tensor ``like`` (deterministic per seed, idea, step and salt)."""
        import torch

        g = torch.Generator(device=like.device).manual_seed(self._key(salt))
        return torch.randn(like.shape, generator=g, device=like.device, dtype=torch.float32).to(like.dtype)

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self._key(salt))


def parse_ideas(spec: str | Sequence[str] | None) -> list[tuple[str, float]]:
    """``"a, b=0.5"`` (or a list of such items) -> ``[("a", 1.0), ("b", 0.5)]``; unknown names raise with a hint."""
    items = spec.split(",") if isinstance(spec, str) else list(spec or [])
    out: list[tuple[str, float]] = []
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in IDEAS:
            near = difflib.get_close_matches(name, IDEAS, n=3)
            hint = f" (did you mean {', '.join(near)}?)" if near else " (see --list-ideas)"
            raise ValueError(f"unknown idea {name!r}{hint}")
        out.append((name, float(value) if value.strip() else 1.0))
    return out


class IdeaStack:
    """The enabled ideas of one run, applied in the order given; falsy when none are enabled."""

    def __init__(self, spec: str | Sequence[str] | None, *, seed: int = 0, native: int = 0, prompt: str = ""):
        self.items = [
            (IDEAS[name], Ctx(name, strength, int(seed), int(native), str(prompt or "")))
            for name, strength in parse_ideas(spec)
            if strength != 0.0
        ]

    def __bool__(self) -> bool:
        return bool(self.items)

    def has(self, hook: str) -> bool:
        return any(idea.hook == hook for idea, _ in self.items)

    def names(self, hook: str | None = None) -> list[str]:
        return [idea.name for idea, _ in self.items if hook is None or idea.hook == hook]

    def begin(self, n_steps: int, shape: Sequence[int]) -> None:
        """Start a sampling pass: reset step position and per-idea state (EMA buffers, anchors)."""
        for _, ctx in self.items:
            ctx.n, ctx.i, ctx.shape = max(1, int(n_steps)), 0, tuple(int(s) for s in shape)
            ctx.state.clear()

    def at(self, i: int) -> None:
        for _, ctx in self.items:
            ctx.i = int(i)

    def _run(self, hook: str, value: Any, *extra: Any) -> Any:
        for idea, ctx in self.items:
            if idea.hook == hook:
                value = idea.fn(value, *extra, ctx)
        return value

    def noise(self, x):
        return self._run("noise", x)

    def scale(self, w: float) -> float:
        return float(self._run("scale", float(w)))

    def delta(self, delta, d_cond, w: float):
        return self._run("delta", delta, d_cond, float(w))

    def x0(self, x0):
        return self._run("x0", x0)

    def step(self, x_new, x_prev, x0):
        return self._run("step", x_new, x_prev, x0)

    def image(self, img: np.ndarray) -> np.ndarray:
        """``img`` in [0, 255] (float or uint8, RGB or RGBA) -> same layout; uint8 when a dither ran last."""
        if not self.has("image"):
            return img
        alpha = img[..., 3:] if img.shape[-1] == 4 else None
        rgb = np.clip(np.asarray(img[..., :3], dtype=np.float32) / 255.0, 0.0, 1.0)
        rgb = np.clip(self._run("image", rgb), 0.0, 1.0).astype(np.float32) * 255.0
        last = [idea for idea, _ in self.items if idea.hook == "image"][-1]
        if last.quantizes or img.dtype == np.uint8:
            rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        if alpha is not None:
            rgb = np.concatenate([rgb.astype(img.dtype), alpha], axis=-1)
        return rgb


def idea_stack_from_args(args: Any, *, native: int = 0) -> IdeaStack:
    return IdeaStack(
        getattr(args, "ideas", "") or "",
        seed=int(getattr(args, "seed", 0) or 0),
        native=native,
        prompt=str(getattr(args, "prompt", "") or ""),
    )


def apply_image_ideas(args: Any, images: list) -> list:
    """The ``image``-hook ideas of ``--ideas`` on every final image (seeded per candidate)."""
    stack = idea_stack_from_args(args)
    if not stack.has("image"):
        return images
    out = []
    for k, img in enumerate(images):
        for _, ctx in stack.items:
            ctx.seed = int(getattr(args, "seed", 0) or 0) + k
        out.append(stack.image(img))
    return out


def ideas_table(hook: str | None = None) -> str:
    """Plain-text catalogue for ``--list-ideas``."""
    rows = [i for i in IDEAS.values() if hook is None or i.hook == hook]
    width = max(len(i.name) for i in rows)
    lines = [f"{len(rows)} ideas (enable with --ideas name[=strength],...; strength 0 = off)"]
    for h in HOOKS:
        group = [i for i in rows if i.hook == h]
        if group:
            lines.append(f"\n[{h}]")
            lines += [f"  {i.name:<{width}}  {i.summary}" for i in group]
    return "\n".join(lines)


from utils.ideas import image, latent  # noqa: E402,F401  (registers the ideas)
