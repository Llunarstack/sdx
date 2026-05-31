"""
ControlNet scaffold registry and lightweight control-map extractors.

Heavy ControlNet inference stays in Diffusers pipelines; this module tracks
which ``pretrained/`` folders resolve and builds cheap PIL-based proxies
(canny, softedge) when neural extractors are unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image, ImageFilter

from utils.modeling import hf_scaffold
from utils.modeling.model_paths import (
    default_controlnet_canny_path,
    default_controlnet_canny_sdxl_path,
    default_controlnet_depth_path,
    default_controlnet_depth_sdxl_path,
    default_controlnet_hed_path,
    default_controlnet_lineart_path,
    default_controlnet_mlsd_path,
    default_controlnet_normal_path,
    default_controlnet_openpose_path,
    default_controlnet_openpose_sdxl_path,
    default_controlnet_scribble_path,
    default_controlnet_seg_path,
    default_controlnet_softedge_path,
    default_controlnet_union_sdxl_path,
)

ControlPathFn = Callable[[], str]

CONTROLNET_REGISTRY: Dict[str, ControlPathFn] = {
    "canny": default_controlnet_canny_path,
    "depth": default_controlnet_depth_path,
    "openpose": default_controlnet_openpose_path,
    "lineart": default_controlnet_lineart_path,
    "scribble": default_controlnet_scribble_path,
    "mlsd": default_controlnet_mlsd_path,
    "softedge": default_controlnet_softedge_path,
    "seg": default_controlnet_seg_path,
    "normal": default_controlnet_normal_path,
    "hed": default_controlnet_hed_path,
    "canny_sdxl": default_controlnet_canny_sdxl_path,
    "depth_sdxl": default_controlnet_depth_sdxl_path,
    "openpose_sdxl": default_controlnet_openpose_sdxl_path,
    "union_sdxl": default_controlnet_union_sdxl_path,
}

# Maps that can be synthesized without neural weights.
PIL_PROXY_TYPES: frozenset[str] = frozenset({"canny", "softedge", "hed"})


def list_controlnet_types() -> List[str]:
    return sorted(CONTROLNET_REGISTRY.keys())


def resolve_controlnet_path(control_type: str) -> str:
    key = str(control_type).strip().lower()
    fn = CONTROLNET_REGISTRY.get(key)
    if fn is None:
        raise KeyError(f"Unknown control type: {control_type}")
    return fn()


def controlnet_has_weights(control_type: str) -> bool:
    try:
        path = resolve_controlnet_path(control_type)
        return hf_scaffold.has_local_weights(path)
    except KeyError:
        return False


def extract_pil_proxy(path: str, output_path: str, control_type: str) -> str:
    """Write a cheap proxy control map (canny or softedge)."""
    kind = str(control_type).strip().lower()
    if kind not in PIL_PROXY_TYPES:
        return ""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    if kind == "canny":
        out = img.convert("L").filter(ImageFilter.FIND_EDGES)
    elif kind == "hed":
        out = img.convert("L").filter(ImageFilter.EMBOSS).filter(ImageFilter.FIND_EDGES)
    else:
        out = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1)).filter(ImageFilter.FIND_EDGES)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest)
    return str(dest)


def extract_control_maps_batch(
    path: str,
    output_dir: Path,
    *,
    types: Sequence[str] = ("canny", "softedge", "depth"),
    device: str = "cuda",
) -> Dict[str, str]:
    """Merge PIL proxies + HF depth/normal loaders."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    maps: Dict[str, str] = {}

    for kind in types:
        key = str(kind).strip().lower()
        if key in PIL_PROXY_TYPES:
            dest = out_dir / f"{stem}_{key}.png"
            out = extract_pil_proxy(path, str(dest), key)
            if out:
                maps[key] = out
            continue
        if key == "depth":
            try:
                from utils.modeling.hf_loaders import depth_map, marigold_depth_map

                dest = out_dir / f"{stem}_depth.png"
                for prefer in ("small", "base", "large", "dpt", "zoe"):
                    out = depth_map(str(path), str(dest), device=device, prefer=prefer)
                    if out:
                        maps["depth"] = out
                        break
                if "depth" not in maps:
                    out = marigold_depth_map(str(path), str(dest), device=device)
                    if out:
                        maps["depth"] = out
            except Exception:
                pass
        elif key == "normals":
            try:
                from utils.modeling.hf_loaders import normals_map

                dest = out_dir / f"{stem}_normals.png"
                out = normals_map(str(path), str(dest), device=device)
                if out:
                    maps["normals"] = out
            except Exception:
                pass

    return maps


__all__ = [
    "CONTROLNET_REGISTRY",
    "PIL_PROXY_TYPES",
    "controlnet_has_weights",
    "extract_control_maps_batch",
    "extract_pil_proxy",
    "list_controlnet_types",
    "resolve_controlnet_path",
]
