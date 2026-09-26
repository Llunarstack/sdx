"""
Multimodal ref bus — Seedance 2.5 / Hailuo H3 style role-tagged references.

Leaders accept up to ~50 mixed refs with roles (identity, product, style, camera,
voice, motion, layout). We unify image/video/audio into one conditioned pack
the pipeline can consume without guessing order-based weighting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "RefRole",
    "ModalityRef",
    "MultimodalRefPack",
    "parse_multimodal_refs",
    "compile_ref_prompt_garnish",
    "resolve_identity_paths",
    "resolve_audio_paths",
    "resolve_motion_video_paths",
    "ref_budget_ok",
]

RefRole = Literal[
    "identity",
    "product",
    "style",
    "camera",
    "voice",
    "motion",
    "layout",
    "scene",
    "wardrobe",
]


@dataclass(slots=True)
class ModalityRef:
    path: str
    role: RefRole = "identity"
    modality: Literal["image", "video", "audio"] = "image"
    strength: float = 0.85
    tag: str = ""
    notes: str = ""


@dataclass(slots=True)
class MultimodalRefPack:
    """Hailuo/Seedance-compatible reference budget."""

    refs: list[ModalityRef] = field(default_factory=list)
    max_images: int = 30
    max_videos: int = 10
    max_audios: int = 10

    def by_role(self, role: str) -> list[ModalityRef]:
        return [r for r in self.refs if r.role == role]

    def by_modality(self, modality: str) -> list[ModalityRef]:
        return [r for r in self.refs if r.modality == modality]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "image": len(self.by_modality("image")),
            "video": len(self.by_modality("video")),
            "audio": len(self.by_modality("audio")),
            "total": len(self.refs),
        }


_EXT_MOD = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".aac": "audio",
    ".m4a": "audio",
    ".flac": "audio",
}


def _guess_modality(path: str) -> str:
    return _EXT_MOD.get(Path(path).suffix.lower(), "image")


def _coerce_role(raw: str) -> RefRole:
    key = (raw or "identity").strip().lower().replace("-", "_")
    aliases = {
        "character": "identity",
        "face": "identity",
        "subject": "identity",
        "sku": "product",
        "logo": "product",
        "look": "style",
        "aesthetic": "style",
        "cam": "camera",
        "lens": "camera",
        "speech": "voice",
        "sfx": "voice",
        "music": "voice",
        "action": "motion",
        "mocap": "motion",
        "blockout": "layout",
        "3d": "layout",
        "location": "scene",
        "set": "scene",
        "outfit": "wardrobe",
        "costume": "wardrobe",
    }
    key = aliases.get(key, key)
    allowed: tuple[RefRole, ...] = (
        "identity",
        "product",
        "style",
        "camera",
        "voice",
        "motion",
        "layout",
        "scene",
        "wardrobe",
    )
    return key if key in allowed else "identity"  # type: ignore[return-value]


def parse_multimodal_refs(raw: Any) -> MultimodalRefPack:
    """
    Accepts:
      - list of paths / dicts
      - dict with keys images/videos/audios or role buckets
      - Seedance-like ``references: [{path, role, ...}]``
    """
    pack = MultimodalRefPack()
    if raw is None:
        return pack
    items: list[Any] = []
    if isinstance(raw, Mapping):
        if "references" in raw or "refs" in raw:
            items = list(raw.get("references") or raw.get("refs") or [])
        else:
            for key in ("images", "image", "videos", "video", "audios", "audio"):
                val = raw.get(key)
                if isinstance(val, str):
                    items.append({"path": val, "role": "identity" if "image" in key else key.rstrip("s")})
                elif isinstance(val, Sequence):
                    for v in val:
                        if isinstance(v, str):
                            items.append({"path": v, "role": "identity" if "image" in key else key.rstrip("s")})
                        elif isinstance(v, Mapping):
                            items.append(v)
            # Role buckets: {"identity": [...], "style": [...]}
            for role_key, val in raw.items():
                if role_key in ("images", "videos", "audios", "references", "refs", "image", "video", "audio"):
                    continue
                if isinstance(val, (str, Mapping)):
                    val = [val]
                if isinstance(val, Sequence):
                    for v in val:
                        if isinstance(v, str):
                            items.append({"path": v, "role": role_key})
                        elif isinstance(v, Mapping):
                            d = dict(v)
                            d.setdefault("role", role_key)
                            items.append(d)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = list(raw)
    else:
        return pack

    for it in items:
        if isinstance(it, str):
            path = it.strip()
            if not path:
                continue
            pack.refs.append(
                ModalityRef(path=path, role="identity", modality=_guess_modality(path))  # type: ignore[arg-type]
            )
            continue
        if not isinstance(it, Mapping):
            continue
        path = str(it.get("path") or it.get("url") or it.get("file") or it.get("src") or "").strip()
        if not path:
            continue
        modality = str(it.get("modality") or it.get("type") or _guess_modality(path)).lower()
        if modality not in ("image", "video", "audio"):
            modality = _guess_modality(path)
        pack.refs.append(
            ModalityRef(
                path=path,
                role=_coerce_role(str(it.get("role") or it.get("kind") or "identity")),
                modality=modality,  # type: ignore[arg-type]
                strength=float(it.get("strength") or it.get("weight") or 0.85),
                tag=str(it.get("tag") or it.get("id") or ""),
                notes=str(it.get("notes") or it.get("hint") or ""),
            )
        )
    return pack


def ref_budget_ok(pack: MultimodalRefPack) -> tuple[bool, list[str]]:
    """Seedance 2.5-ish budgets; Soft-warn rather than hard-fail."""
    c = pack.counts
    issues = []
    if c["image"] > pack.max_images:
        issues.append(f"images={c['image']}>{pack.max_images}")
    if c["video"] > pack.max_videos:
        issues.append(f"videos={c['video']}>{pack.max_videos}")
    if c["audio"] > pack.max_audios:
        issues.append(f"audios={c['audio']}>{pack.max_audios}")
    if c["total"] > 50:
        issues.append(f"total={c['total']}>50")
    return len(issues) == 0, issues


def compile_ref_prompt_garnish(pack: MultimodalRefPack) -> tuple[str, str]:
    """Positive/negative fragments describing bound roles (for keyframe prompts)."""
    pos_bits: list[str] = []
    neg_bits: list[str] = []
    for role in ("identity", "product", "wardrobe", "style", "scene", "camera", "motion"):
        rs = pack.by_role(role)
        if not rs:
            continue
        tags = ", ".join(r.tag or Path(r.path).stem for r in rs[:4])
        pos_bits.append(f"{role} locked to refs [{tags}]")
    if pack.by_role("identity"):
        neg_bits.append("identity drift, face morph, wardrobe change")
    if pack.by_role("product"):
        neg_bits.append("logo melt, warped label text, SKU morph")
    if pack.by_role("voice"):
        pos_bits.append("lip sync to voice reference")
    return "; ".join(pos_bits), ", ".join(neg_bits)


def resolve_identity_paths(pack: MultimodalRefPack) -> list[str]:
    out: list[str] = []
    for r in pack.by_role("identity") + pack.by_role("wardrobe") + pack.by_role("product"):
        if r.modality == "image" and Path(r.path).is_file():
            out.append(r.path)
    return out


def resolve_audio_paths(pack: MultimodalRefPack) -> list[str]:
    return [r.path for r in pack.by_modality("audio") if Path(r.path).is_file() or r.path.startswith("http")]


def resolve_motion_video_paths(pack: MultimodalRefPack) -> list[str]:
    vids = pack.by_role("motion") + pack.by_role("camera")
    return [r.path for r in vids if r.modality == "video"]
