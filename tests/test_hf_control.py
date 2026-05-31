from pathlib import Path

from utils.modeling.hf_control import (
    CONTROLNET_REGISTRY,
    extract_pil_proxy,
    list_controlnet_types,
    resolve_controlnet_path,
)


def test_controlnet_registry_has_extended_types():
    types = list_controlnet_types()
    for name in ("mlsd", "softedge", "seg", "normal", "hed", "canny_sdxl"):
        assert name in types
    assert len(CONTROLNET_REGISTRY) >= 12


def test_resolve_controlnet_path_returns_string():
    p = resolve_controlnet_path("canny")
    assert isinstance(p, str)
    assert p


def test_pil_softedge_proxy(tmp_path: Path):
    from PIL import Image

    src = tmp_path / "in.png"
    out = tmp_path / "soft.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(src)
    written = extract_pil_proxy(str(src), str(out), "softedge")
    assert written == str(out)
    assert out.is_file()
