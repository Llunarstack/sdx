"""Per-region condition annealing (CADS-style) for box layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class PerRegionCADSConfig:
    """Anneal regional condition noise early, global late."""

    region_strengths: tuple[float, ...]
    global_strength: float = 1.0
    anneal_power: float = 1.2

    @classmethod
    def from_box_spec(cls, box_spec: Any, *, base: float = 0.35) -> PerRegionCADSConfig:
        n = len(getattr(box_spec, "regions", []) or [])
        strengths = tuple(float(base) for _ in range(max(n, 1)))
        return cls(region_strengths=strengths, global_strength=1.0)

    def regional_noise_scale(self, step_index: int, total_steps: int, region_index: int) -> float:
        if region_index < 0 or region_index >= len(self.region_strengths):
            return 1.0
        progress = step_index / max(total_steps - 1, 1)
        early = (1.0 - progress) ** self.anneal_power
        return 1.0 + early * self.region_strengths[region_index]


def merge_cads_into_holy_grail(holy_kw: Mapping[str, Any], cfg: Optional[PerRegionCADSConfig]) -> dict:
    """Boost holy-grail CADS when per-region config is active."""
    out = dict(holy_kw)
    if cfg is None:
        return out
    if float(out.get("holy_grail_cads_strength", 0.0) or 0.0) <= 0.0:
        out["holy_grail_cads_strength"] = 0.15
    return out
