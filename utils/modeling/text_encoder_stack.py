"""
Text encoder stack readiness for triple / penta training and sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

from utils.modeling.hf_scaffold import has_local_weights
from utils.modeling.model_paths import (
    default_clip_bigg_path,
    default_clip_h14_path,
    default_clip_l_path,
    default_longclip_l_path,
    default_t5_path,
    model_dir,
    resolve_model_path,
)

StackMode = Literal["t5", "triple", "penta"]

TRIPLE_CATALOG = ("T5-XXL", "CLIP-ViT-L-14", "CLIP-ViT-bigG-14")
PENTA_CATALOG = TRIPLE_CATALOG + ("CLIP-ViT-H-14", "LongCLIP-L")

_CATALOG_FALLBACK = {
    "T5-XXL": ("google/t5-v1_1-xxl", default_t5_path),
    "CLIP-ViT-L-14": ("openai/clip-vit-large-patch14", default_clip_l_path),
    "CLIP-ViT-bigG-14": ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", default_clip_bigg_path),
    "CLIP-ViT-H-14": ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", default_clip_h14_path),
    "LongCLIP-L": ("creative-graphic-design/LongCLIP-L", default_longclip_l_path),
}


@dataclass(slots=True)
class EncoderSlotStatus:
    name: str
    resolved: str
    local: bool
    has_weights: bool

    @property
    def ready(self) -> bool:
        return self.has_weights or not self.local


@dataclass(slots=True)
class TextEncoderStackStatus:
    mode: StackMode
    slots: List[EncoderSlotStatus]

    @property
    def ready(self) -> bool:
        return all(s.ready for s in self.slots)

    @property
    def local_count(self) -> int:
        return sum(1 for s in self.slots if s.local)

    @property
    def weights_count(self) -> int:
        return sum(1 for s in self.slots if s.has_weights)


def catalog_for_mode(mode: str) -> tuple[str, ...]:
    m = str(mode or "t5").lower()
    if m == "penta":
        return PENTA_CATALOG
    if m == "triple":
        return TRIPLE_CATALOG
    return ("T5-XXL",)


def _slot_status(name: str) -> EncoderSlotStatus:
    hub_id, default_fn = _CATALOG_FALLBACK[name]
    resolved = resolve_model_path(name, hub_id)
    local_path = Path(resolved)
    is_local = local_path.is_absolute() and local_path.is_dir() and any(local_path.iterdir())
    if not is_local:
        local_path = model_dir() / name
        is_local = local_path.is_dir() and any(local_path.iterdir())
        if is_local:
            resolved = str(local_path)
    has_w = has_local_weights(Path(resolved)) if is_local else False
    if not has_w and not is_local:
        resolved = default_fn()
    return EncoderSlotStatus(name=name, resolved=resolved, local=is_local, has_weights=has_w)


def stack_status(mode: str = "penta") -> TextEncoderStackStatus:
    """Return per-encoder local/weights status for ``t5``, ``triple``, or ``penta``."""
    m = str(mode or "t5").lower()
    if m not in ("t5", "triple", "penta"):
        m = "t5"
    names = catalog_for_mode(m)
    return TextEncoderStackStatus(mode=m, slots=[_slot_status(n) for n in names])


def stack_status_lines(mode: str = "penta") -> List[str]:
    """Markdown bullet lines for eval reports and CLI."""
    st = stack_status(mode)
    if st.mode == "t5":
        s = st.slots[0]
        w = "weights" if s.has_weights else ("config-only" if s.local else "hub fallback")
        return [f"- T5: `{s.resolved}` ({w})"]

    lines = [
        f"- Mode: **{st.mode}** — {st.weights_count}/{len(st.slots)} with local weights, "
        f"{st.local_count}/{len(st.slots)} local folders"
    ]
    for s in st.slots:
        if s.has_weights:
            tag = "weights"
        elif s.local:
            tag = "config-only"
        else:
            tag = "hub fallback"
        lines.append(f"  - {s.name}: `{s.resolved}` ({tag})")
    if not st.ready:
        missing = [s.name for s in st.slots if s.local and not s.has_weights]
        if missing:
            lines.append(
                f"- Note: config-only locally ({', '.join(missing)}); runtime uses HF hub unless weights downloaded."
            )
    return lines


def stack_download_hint(mode: str = "penta") -> str:
    m = str(mode or "t5").lower()
    if m == "penta":
        return (
            "python scripts/download/download_models.py --penta-text-encoders\n"
            "python scripts/download/download_hf_scaffold.py --penta"
        )
    if m == "triple":
        return "python scripts/download/download_models.py --triple-text-encoders"
    return "python scripts/download/download_models.py --t5"


__all__ = [
    "EncoderSlotStatus",
    "PENTA_CATALOG",
    "TRIPLE_CATALOG",
    "TextEncoderStackStatus",
    "catalog_for_mode",
    "stack_download_hint",
    "stack_status",
    "stack_status_lines",
]
