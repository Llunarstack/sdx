"""

Visual-design helpers: STEM/textbook/UI/brand/architecture and extensions (editorial, decks,

CAD-style flats, fashion flats). Wired from ``sample.py``, book pipeline, multimodal GenerationRequest,

and ``scripts.cli`` generate.



Public names load on first access (submodules and re-exports from sibling modules).

"""



from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [

    "VisualDesignDomain",

    "VisualDesignApplyResult",

    "presets",

    "apply_visual_design_preset_to_prompt",

    "apply_visual_design_pack",

    "apply_visual_design_stage",

    "assert_visual_design_registry_valid",

    "build_visual_design_prompt_pair",

    "design_pack_ids",

    "DOMAIN_NEGATIVES",

    "DOMAIN_POSITIVES",

    "extend_sample_argv_visual_design",

    "merge_negative_addon",

    "merge_visual_fragments",

    "normalize_intensity",

    "preset_ids",

    "prompt_suggests_domain",

    "resolve_visual_design_preset",

    "validate_visual_design_registry",

    "VISUAL_DESIGN_PRESETS",

    "visual_design_cli_domain_choices",

]



# Map each ``__all__`` name to the submodule stem (``*.py`` under this package).

_ATTR_ORIGIN: dict[str, str] = {

    "assert_visual_design_registry_valid": "validate",

    "apply_visual_design_pack": "compose",

    "apply_visual_design_preset_to_prompt": "presets",

    "apply_visual_design_stage": "sampling",

    "build_visual_design_prompt_pair": "compose",

    "design_pack_ids": "compose",

    "DOMAIN_NEGATIVES": "registry",

    "DOMAIN_POSITIVES": "registry",

    "extend_sample_argv_visual_design": "argv",

    "merge_negative_addon": "negatives",

    "merge_visual_fragments": "compose",

    "normalize_intensity": "sampling",

    "preset_ids": "presets",

    "prompt_suggests_domain": "compose",

    "resolve_visual_design_preset": "presets",

    "validate_visual_design_registry": "validate",

    "visual_design_cli_domain_choices": "compose",

    "VisualDesignApplyResult": "sampling",

    "VisualDesignDomain": "compose",

    "VISUAL_DESIGN_PRESETS": "presets",

}





def __getattr__(name: str) -> Any:

    if name not in __all__:

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "presets":

        mod = import_module(".presets", __package__)

        globals()[name] = mod

        return mod

    stem = _ATTR_ORIGIN[name]

    mod = import_module(f".{stem}", __package__)

    val = getattr(mod, name)

    globals()[name] = val

    return val





def __dir__() -> list[str]:

    return sorted(set(globals()) | set(__all__))


