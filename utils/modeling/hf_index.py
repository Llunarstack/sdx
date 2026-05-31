"""HF scaffold index helpers for status reports and tooling."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from utils.modeling.hf_scaffold import HFModelEntry, has_local_weights, scaffold_registry
from utils.modeling.model_paths import model_dir


def role_counts() -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for e in scaffold_registry():
        counts[e.role] += 1
    return dict(sorted(counts.items()))


def local_status_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for e in scaffold_registry():
        fp = model_dir() / e.name
        exists = fp.is_dir() and any(fp.iterdir())
        weights = has_local_weights(fp) if exists else False
        rows.append(
            {
                "name": e.name,
                "role": e.role,
                "repo_id": e.repo_id,
                "exists": bool(exists),
                "has_weights": bool(weights),
                "config_only": bool(exists and not weights),
            }
        )
    return rows


def summary() -> Dict[str, object]:
    rows = local_status_rows()
    return {
        "total_registry": len(scaffold_registry()),
        "role_counts": role_counts(),
        "local_folders": sum(1 for r in rows if r["exists"]),
        "with_weights": sum(1 for r in rows if r["has_weights"]),
        "config_only": sum(1 for r in rows if r["config_only"]),
    }


def models_by_role(role: str) -> List[HFModelEntry]:
    return [e for e in scaffold_registry() if e.role == role]


__all__ = ["local_status_rows", "models_by_role", "role_counts", "summary"]
