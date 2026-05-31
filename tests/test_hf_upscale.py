from utils.modeling.hf_upscale import list_face_restore_models, list_upscale_models


def test_upscale_registry():
    assert "Real-ESRGAN" in list_upscale_models()
    assert "SwinIR-classical-x2" in list_upscale_models()
    assert "CodeFormer" in list_face_restore_models()
    assert "GFPGAN" in list_face_restore_models()
