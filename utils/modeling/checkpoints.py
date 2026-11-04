"""DiT checkpoint loading, checkpoint diffs / LoRA extraction, and the LoRA bank registry."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from utils.prompt.text import normalize_artist_tag
from utils.runtime import to_plain_dict

if TYPE_CHECKING:
    from models.conditioning import RAELatentBridge


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    """``torch.load`` on CPU, memory-mapped when the file is a zip checkpoint (no full read up front)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be mmapped
        return torch.load(path, map_location="cpu", weights_only=False)


def load_dit_text_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    *,
    reject_enhanced: bool = False,
) -> tuple[torch.nn.Module, object, RAELatentBridge | None, str, dict | None]:
    """
    Load a DiT-Text checkpoint and optional RAE latent bridge.

    Returns: (model, config, rae_bridge_or_none, model_name, text_encoder_fusion_sd_or_none)
    """
    from config import get_dit_build_kwargs
    from models import DiT_models_text
    from models.conditioning import RAELatentBridge

    checkpoint_data = load_checkpoint_file(ckpt_path)
    config = checkpoint_data.get("config")
    if config is None:
        raise ValueError("Checkpoint must contain config")

    model_name = getattr(config, "model_name", "DiT-XL/2-Text")
    if reject_enhanced and str(model_name).startswith("EnhancedDiT"):
        raise ValueError(
            "This checkpoint is an EnhancedDiT model, which is not compatible with sample.py (DiT-Text/T5 sampler).\n"
            "Use the EnhancedDiT sampler instead:\n"
            '  python -m scripts.tools enhanced sample "<prompt>" --checkpoint <path_to_ckpt> --output out.png\n'
            "Or train a DiT-*-Text checkpoint with train.py to use sample.py."
        )

    model_builder = DiT_models_text.get(model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_builder(**get_dit_build_kwargs(config, class_dropout_prob=0.0))
    use_ema = bool(checkpoint_data.get("ema"))
    model.load_state_dict(checkpoint_data["ema"] if use_ema else checkpoint_data.get("model"), strict=True)
    model = model.to(device).eval()

    def _aux(key: str):
        """The EMA copy of an auxiliary module trained with the DiT (paired with EMA weights), else its live state."""
        return (checkpoint_data.get(f"ema_{key}") if use_ema else None) or checkpoint_data.get(key)

    rae_bridge = None
    rae_bridge_state_dict = _aux("rae_latent_bridge")
    if isinstance(rae_bridge_state_dict, dict) and rae_bridge_state_dict.get("to_dit.weight") is not None:
        rae_channels = int(rae_bridge_state_dict["to_dit.weight"].shape[1])
        rae_bridge = RAELatentBridge(rae_channels, 4)
        rae_bridge.load_state_dict(rae_bridge_state_dict, strict=True)
        rae_bridge = rae_bridge.to(device).eval()

    fusion_state_dict = _aux("text_encoder_fusion")
    if not isinstance(fusion_state_dict, dict):
        fusion_state_dict = None
    return model, config, rae_bridge, str(model_name), fusion_state_dict


def finetuned_checkpoint(model, config, rae_bridge=None, fusion_sd: dict | None = None, **extra) -> dict:
    """Checkpoint of a model fine-tuned from :func:`load_dit_text_checkpoint` output (weights double as ``ema``),
    keeping the RAE bridge and text-encoder fusion it was loaded with so inference still finds them."""
    ckpt = {"model": model.state_dict(), "ema": model.state_dict(), "config": config, **extra}
    if rae_bridge is not None:
        ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
    if fusion_sd:
        ckpt["text_encoder_fusion"] = fusion_sd
    return ckpt


def load_sampler_checkpoint(
    ckpt_path: str,
    device: str = "cuda",
    *,
    reject_enhanced: bool = True,
    verbose: bool = True,
) -> tuple[torch.nn.Module, object, RAELatentBridge | None, dict | None]:
    """
    Load a checkpoint for ``sample.py`` / inference utilities.

    Returns ``(model, config, rae_bridge, text_encoder_fusion_sd)``.
    """
    model, cfg, rae_bridge, model_name, fusion_sd = load_dit_text_checkpoint(
        ckpt_path,
        device=device,
        reject_enhanced=reject_enhanced,
    )
    if verbose:
        print(f"Model: {model_name}")
        if rae_bridge is not None:
            rae_c = int(rae_bridge.to_dit.weight.shape[1])
            print(f"Loaded RAELatentBridge: rae_channels={rae_c} -> 4 (DiT latent space)")
    return model, cfg, rae_bridge, fusion_sd


def _diff_std_population(diff: torch.Tensor) -> float:
    """Std of absolute diffs per parameter; stable for tiny tensors (avoids unbiased-std warnings)."""
    flat = torch.flatten(diff)
    n = int(flat.numel())
    if n <= 1:
        return 0.0
    return float(torch.std(flat, unbiased=False))


def analyze_checkpoint_differences(checkpoint1_path: str, checkpoint2_path: str) -> dict[str, Any]:
    """Analyze differences between two checkpoints."""
    cp1 = load_checkpoint_file(checkpoint1_path)
    cp2 = load_checkpoint_file(checkpoint2_path)

    analysis: dict[str, Any] = {"parameter_differences": {}, "statistics": {}, "config_differences": {}}

    raw1, raw2 = cp1.get("model"), cp2.get("model")
    model1: dict[str, Any] = raw1 if isinstance(raw1, dict) else {}
    model2: dict[str, Any] = raw2 if isinstance(raw2, dict) else {}

    total_diff = 0.0
    total_params = 0
    max_diff = 0.0
    max_diff_param = ""

    n_tensors_compared = 0
    n_only_checkpoint1 = 0
    n_only_checkpoint2 = 0
    n_shape_mismatch = 0
    n_skipped_non_tensor = 0

    if not isinstance(raw1, dict) and raw1 is not None:
        analysis["parameter_differences"]["__note__"] = {
            "checkpoint1_model": "not_a_dict_skipped",
        }
    if not isinstance(raw2, dict) and raw2 is not None:
        note = analysis["parameter_differences"].setdefault("__note__", {})
        note["checkpoint2_model"] = "not_a_dict_skipped"

    all_keys = sorted(set(model1.keys()) | set(model2.keys()))
    for key in all_keys:
        t1 = model1.get(key)
        t2 = model2.get(key)
        if t1 is None:
            analysis["parameter_differences"][key] = {
                "status": "only_in_checkpoint2",
                "shape": list(t2.shape) if hasattr(t2, "shape") else [],
            }
            n_only_checkpoint2 += 1
            continue
        if t2 is None:
            analysis["parameter_differences"][key] = {
                "status": "only_in_checkpoint1",
                "shape": list(t1.shape) if hasattr(t1, "shape") else [],
            }
            n_only_checkpoint1 += 1
            continue
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor):
            analysis["parameter_differences"][key] = {
                "status": "skipped_non_tensor",
                "type_checkpoint1": type(t1).__name__,
                "type_checkpoint2": type(t2).__name__,
            }
            n_skipped_non_tensor += 1
            continue
        if t1.shape != t2.shape:
            analysis["parameter_differences"][key] = {
                "status": "shape_mismatch",
                "shape_checkpoint1": list(t1.shape),
                "shape_checkpoint2": list(t2.shape),
            }
            n_shape_mismatch += 1
            continue

        diff = torch.abs(t1 - t2)
        param_diff = torch.mean(diff).item()
        param_max_diff = torch.max(diff).item()

        analysis["parameter_differences"][key] = {
            "mean_diff": param_diff,
            "max_diff": param_max_diff,
            "std_diff": _diff_std_population(diff),
            "shape": list(t1.shape),
        }

        total_diff += param_diff * t1.numel()
        total_params += t1.numel()
        n_tensors_compared += 1

        if param_max_diff > max_diff:
            max_diff = param_max_diff
            max_diff_param = key

    if not raw1:
        mc1 = "missing_or_empty"
    elif not isinstance(raw1, dict):
        mc1 = "wrong_type"
    else:
        mc1 = "ok"
    if not raw2:
        mc2 = "missing_or_empty"
    elif not isinstance(raw2, dict):
        mc2 = "wrong_type"
    else:
        mc2 = "ok"

    # Overall statistics
    analysis["statistics"] = {
        "total_parameters": total_params,
        "average_difference": total_diff / total_params if total_params > 0 else 0.0,
        "max_difference": max_diff,
        "max_difference_parameter": max_diff_param,
        "step_difference": cp2.get("step", 0) - cp1.get("step", 0),
        "loss_difference": cp2.get("loss", 0) - cp1.get("loss", 0),
        "model_dict_checkpoint1": mc1,
        "model_dict_checkpoint2": mc2,
        "tensors_compared": n_tensors_compared,
        "tensors_only_checkpoint1": n_only_checkpoint1,
        "tensors_only_checkpoint2": n_only_checkpoint2,
        "tensors_shape_mismatch": n_shape_mismatch,
        "tensors_skipped_non_tensor": n_skipped_non_tensor,
        "tensor_keys_checked": len(all_keys),
    }

    # Compare configs if available
    if "config" in cp1 and "config" in cp2:
        config1 = cp1["config"]
        config2 = cp2["config"]

        try:
            config1_dict = to_plain_dict(config1)
            config2_dict = to_plain_dict(config2)

            for key in set(config1_dict.keys()) | set(config2_dict.keys()):
                val1 = config1_dict.get(key, "MISSING")
                val2 = config2_dict.get(key, "MISSING")

                if val1 != val2:
                    analysis["config_differences"][key] = {"checkpoint1": val1, "checkpoint2": val2}
        except Exception:
            analysis["config_differences"] = {"error": "Could not compare configs"}

    cfg_diff = analysis["config_differences"]
    if cfg_diff and "error" not in cfg_diff:
        analysis["statistics"]["config_keys_differing"] = len(cfg_diff)

    return analysis


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
    def from_dict(cls, raw: dict, *, bank_root: Path) -> LoRAEntry | None:
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
    artists: dict[str, LoRAEntry] = field(default_factory=dict)
    styles: dict[str, LoRAEntry] = field(default_factory=dict)
    extras: dict[str, LoRAEntry] = field(default_factory=dict)

    def lookup_artist(self, query: str) -> LoRAEntry | None:
        key = slugify_lora_key(query)
        for bucket in (self.artists, self.extras):
            if key in bucket:
                return bucket[key]
            q = query.strip().lower()
            for k, ent in bucket.items():
                if k == q or k.replace("_", " ") == q.replace("_", " "):
                    return ent
        return None

    def lookup_style(self, query: str) -> LoRAEntry | None:
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
    def load(cls, path: str | os.PathLike[str]) -> LoRABank:
        p = Path(path)
        if not p.is_file():
            return cls(root=default_bank_root())
        raw = json.loads(p.read_text(encoding="utf-8"))
        root = Path(str(raw.get("root") or default_bank_root()))
        bank = cls(root=root)

        def _load_bucket(key: str, dest: dict[str, LoRAEntry]) -> None:
            for k, val in (raw.get(key) or {}).items():
                if isinstance(val, dict):
                    ent = LoRAEntry.from_dict(val, bank_root=root)
                    if ent and Path(ent.lora).is_file():
                        dest[str(k)] = ent

        _load_bucket("artists", bank.artists)
        _load_bucket("styles", bank.styles)
        _load_bucket("extras", bank.extras)
        return bank


def _parse_at_mentions(prompt: str) -> list[str]:
    """Extract raw @artist handles from a prompt (before expansion)."""
    names: list[str] = []
    for m in re.finditer(r"@(?:artist:)?(?P<q>'[^']+'|\"[^\"]+\"|[^\s,+|]+)", prompt):
        raw = m.group("q").strip("'\"")
        if raw:
            names.append(raw)
    return names


def _parse_style_lora_mentions(prompt: str) -> list[tuple[str, float]]:
    """``@style:anime`` or ``@lora:anime:0.6`` → (style_key, scale_mult)."""
    out: list[tuple[str, float]] = []
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
    existing_specs: Sequence[str] | None = None,
) -> list[str]:
    """
    Build ``path:scale:role`` specs from ``@artist`` / ``@style:name`` mentions.

    *artist_strength* scales artist LoRAs; *style_strength* scales style LoRAs.
    Explicit ``--lora`` specs in *existing_specs* are preserved (prepended).
    """
    specs: list[str] = list(existing_specs or [])
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


def augment_sample_lora_args(args: Any) -> list[str]:
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
