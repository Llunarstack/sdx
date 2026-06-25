"""
Provenance bundle — everything needed to reproduce or audit one image.

Ahead of curve for open models: C2PA is coming; local-first audit JSON is cheap now.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class GenerationAudit:
    audit_id: str
    created_unix: float
    prompt: str
    negative_prompt: str
    seed: int
    ckpt_path: str
    args: Dict[str, Any] = field(default_factory=dict)
    frontier_plan: Optional[Dict[str, Any]] = None
    git_commit: str = ""
    python_version: str = ""
    platform: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_audit_bundle(
    *,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    ckpt_path: str = "",
    args: Optional[Mapping[str, Any]] = None,
    frontier_plan: Optional[Any] = None,
    git_commit: str = "",
) -> GenerationAudit:
    fp = _short_hash(f"{prompt}|{seed}|{ckpt_path}|{time.time()}")
    plan_dict: Optional[Dict[str, Any]] = None
    if frontier_plan is not None:
        if hasattr(frontier_plan, "__dataclass_fields__"):
            from dataclasses import asdict as dc_asdict

            plan_dict = dc_asdict(frontier_plan)
        elif isinstance(frontier_plan, dict):
            plan_dict = dict(frontier_plan)

    safe_args: Dict[str, Any] = {}
    if args:
        for k, v in args.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                safe_args[k] = v
            except (TypeError, ValueError):
                safe_args[k] = str(v)

    return GenerationAudit(
        audit_id=fp,
        created_unix=time.time(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        ckpt_path=ckpt_path,
        args=safe_args,
        frontier_plan=plan_dict,
        git_commit=git_commit,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


def write_audit_bundle(audit: GenerationAudit, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    return p


__all__ = ["GenerationAudit", "build_audit_bundle", "write_audit_bundle"]
