"""LoRA bank registry — modular artist/style adapters with user-controlled weights."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.prompt.artist_tag import normalize_artist_tag

__all__ = [
    "LoRABank",
    "LoRAEntry",
    "default_bank_index_path",
    "default_bank_root",
    "resolve_lora_specs_from_prompt",
    "slugify_lora_key",
]

_ROLE_ALIASES = {
    "artist": "style",
    "art": "style",
    "style": "style",
    "character": "character",
    "char": "character",
    "detail": "detail",
    "composition": "composition",
    "medium": "style",
}


def default_bank_root() -> Path:
    data = os.environ.get("SDX_DATA", "").strip()
    if data:
        return Path(data) / "lora_bank"
    return Path(__file__).resolve().parents[2] / "data" / "lora_bank"


def default_bank_index_path() -> Path:
    env = os.environ.get("SDX_LORA_BANK_INDEX", "").strip()
    if env:
        return Path(env)
    return default_bank_root() / "index.json"


def slugify_lora_key(name: str) -> str:
    s = normalize_artist_tag(name).lower()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    return s.strip("_") or "unknown"


@dataclass
class LoRAEntry:
    lora: str
    default_scale: float = 0.75
    role: str = "style"
    trigger: str = ""

    def to_dict(self) -> dict:
        return {
            "lora": self.lora,
            "default_scale": self.default_scale,
            "role": self.role,
            **({"trigger": self.trigger} if self.trigger else {}),
        }

    @classmethod
    def from_dict(cls, raw: dict, *, bank_root: Path) -> Optional["LoRAEntry"]:
        path = str(raw.get("lora") or raw.get("path") or "").strip()
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = bank_root / p
        role = str(raw.get("role") or "style").strip().lower()
        role = _ROLE_ALIASES.get(role, role)
        return cls(
            lora=str(p),
            default_scale=float(raw.get("default_scale", raw.get("scale", 0.75))),
            role=role,
            trigger=str(raw.get("trigger") or "").strip(),
        )


@dataclass
class LoRABank:
    root: Path
    artists: Dict[str, LoRAEntry] = field(default_factory=dict)
    styles: Dict[str, LoRAEntry] = field(default_factory=dict)
    extras: Dict[str, LoRAEntry] = field(default_factory=dict)

    def lookup_artist(self, query: str) -> Optional[LoRAEntry]:
        key = slugify_lora_key(query)
        for bucket in (self.artists, self.extras):
            if key in bucket:
                return bucket[key]
            q = query.strip().lower()
            for k, ent in bucket.items():
                if k == q or k.replace("_", " ") == q.replace("_", " "):
                    return ent
        return None

    def lookup_style(self, query: str) -> Optional[LoRAEntry]:
        key = slugify_lora_key(query)
        return self.styles.get(key) or self.styles.get(query.strip().lower())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "root": str(self.root),
            "artists": {k: v.to_dict() for k, v in sorted(self.artists.items())},
            "styles": {k: v.to_dict() for k, v in sorted(self.styles.items())},
            "extras": {k: v.to_dict() for k, v in sorted(self.extras.items())},
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LoRABank":
        p = Path(path)
        if not p.is_file():
            return cls(root=default_bank_root())
        raw = json.loads(p.read_text(encoding="utf-8"))
        root = Path(str(raw.get("root") or default_bank_root()))
        bank = cls(root=root)

        def _load_bucket(key: str, dest: Dict[str, LoRAEntry]) -> None:
            for k, val in (raw.get(key) or {}).items():
                if isinstance(val, dict):
                    ent = LoRAEntry.from_dict(val, bank_root=root)
                    if ent and Path(ent.lora).is_file():
                        dest[str(k)] = ent

        _load_bucket("artists", bank.artists)
        _load_bucket("styles", bank.styles)
        _load_bucket("extras", bank.extras)
        return bank


def _parse_at_mentions(prompt: str) -> List[str]:
    """Extract raw @artist handles from a prompt (before expansion)."""
    names: List[str] = []
    for m in re.finditer(r"@(?:artist:)?(?P<q>'[^']+'|\"[^\"]+\"|[^\s,+|]+)", prompt):
        raw = m.group("q").strip("'\"")
        if raw:
            names.append(raw)
    return names


def _parse_style_lora_mentions(prompt: str) -> List[Tuple[str, float]]:
    """``@style:anime`` or ``@lora:anime:0.6`` → (style_key, scale_mult)."""
    out: List[Tuple[str, float]] = []
    for m in re.finditer(r"@(?:style|lora):(?P<name>[a-z0-9_-]+)(?::(?P<scale>[0-9.]+))?", prompt, re.I):
        name = m.group("name").strip().lower()
        scale = float(m.group("scale")) if m.group("scale") else 1.0
        out.append((name, scale))
    return out


def resolve_lora_specs_from_prompt(
    prompt: str,
    bank: LoRABank,
    *,
    artist_strength: float = 1.0,
    style_strength: float = 1.0,
    existing_specs: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Build ``path:scale:role`` specs from ``@artist`` / ``@style:name`` mentions.

    *artist_strength* scales artist LoRAs; *style_strength* scales style LoRAs.
    Explicit ``--lora`` specs in *existing_specs* are preserved (prepended).
    """
    specs: List[str] = list(existing_specs or [])
    seen_paths: set[str] = set()
    for spec in specs:
        path = spec.split(":")[0].strip()
        if path:
            seen_paths.add(path)

    for artist in _parse_at_mentions(prompt):
        ent = bank.lookup_artist(artist)
        if ent is None or not Path(ent.lora).is_file():
            continue
        if ent.lora in seen_paths:
            continue
        scale = max(0.0, min(2.0, ent.default_scale * float(artist_strength)))
        specs.append(f"{ent.lora}:{scale:.3f}:{ent.role}")
        seen_paths.add(ent.lora)

    for style_key, mult in _parse_style_lora_mentions(prompt):
        ent = bank.lookup_style(style_key)
        if ent is None or not Path(ent.lora).is_file():
            continue
        if ent.lora in seen_paths:
            continue
        scale = max(0.0, min(2.0, ent.default_scale * float(style_strength) * mult))
        specs.append(f"{ent.lora}:{scale:.3f}:{ent.role}")
        seen_paths.add(ent.lora)

    return specs


def augment_sample_lora_args(args: Any) -> List[str]:
    """
    Mutate ``args.lora`` from bank + prompt. Returns resolved artist names for logging.

    Honors ``--lora-bank`` / ``--no-lora-bank`` and ``SDX_LORA_BANK_INDEX``.
    """
    use_bank = not getattr(args, "no_lora_bank", False)
    if not use_bank:
        return []
    if not getattr(args, "lora_bank", False) and not default_bank_index_path().is_file():
        return []

    index = str(getattr(args, "lora_bank_index", "") or "").strip() or str(default_bank_index_path())
    bank = LoRABank.load(index)
    if not bank.artists and not bank.styles and not bank.extras:
        return []

    prompt = str(getattr(args, "prompt", "") or "")
    raw_prompt = str(getattr(args, "_raw_prompt_before_compose", "") or prompt)
    scan = raw_prompt if "@" in raw_prompt else prompt

    existing = list(getattr(args, "lora", None) or [])
    specs = resolve_lora_specs_from_prompt(
        scan,
        bank,
        artist_strength=float(getattr(args, "artist_strength", 1.0) or 1.0),
        style_strength=float(getattr(args, "style_lora_strength", 1.0) or 1.0),
        existing_specs=existing,
    )
    if specs != existing:
        args.lora = specs
    return _parse_at_mentions(scan)
