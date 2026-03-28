import torch

from models.controlnet import (
    ControlNetEncoder,
    control_type_to_id,
    encode_control_stack,
    infer_control_type_from_path,
    inject_control_tokens,
)


def test_control_encoder_respects_target_hw_and_channels():
    enc = ControlNetEncoder(control_size=32, patch_size=2, in_channels=3, hidden_size=16)
    # Single-channel control map should be promoted to 3 channels.
    ctrl = torch.randn(2, 1, 80, 56)
    out = enc(ctrl, target_hw=(40, 24))
    # target_hw (40,24) with patch_size=2 -> 20*12 tokens
    assert out.shape == (2, 240, 16)


def test_inject_control_tokens_preserves_non_patch_tokens():
    b, patch_n, reg_n, d = 2, 16, 4, 8
    x = torch.randn(b, patch_n + reg_n, d)
    reg_before = x[:, patch_n:, :].clone()
    control = torch.randn(b, patch_n, d)

    y = inject_control_tokens(x, control, control_scale=0.75, patch_tokens=patch_n)

    # Register/non-patch tokens remain untouched.
    assert torch.allclose(y[:, patch_n:, :], reg_before)
    # Patch tokens were modified.
    assert not torch.allclose(y[:, :patch_n, :], x[:, :patch_n, :])


def test_inject_control_tokens_supports_per_batch_tensor_scale():
    b, patch_n, d = 2, 9, 4
    x = torch.randn(b, patch_n, d)
    control = torch.randn(b, patch_n, d)
    scales = torch.tensor([0.0, 1.0], dtype=torch.float32)

    y = inject_control_tokens(x, control, control_scale=scales, patch_tokens=patch_n)

    # Batch 0 unchanged at scale 0; batch 1 changed at scale 1.
    assert torch.allclose(y[0], x[0])
    assert not torch.allclose(y[1], x[1])


def test_control_type_mapping_and_path_inference():
    assert control_type_to_id("edge") == control_type_to_id("canny")
    assert infer_control_type_from_path("foo/bar/depth_map.png") == "depth"
    assert infer_control_type_from_path("foo/bar/openpose.png") == "pose"


def test_control_encoder_type_embedding_changes_features():
    enc = ControlNetEncoder(control_size=32, patch_size=2, in_channels=3, hidden_size=12, num_control_types=9)
    ctrl = torch.randn(1, 3, 32, 32)
    out_a = enc(ctrl, control_type=torch.tensor([control_type_to_id("canny")], dtype=torch.long))
    out_b = enc(ctrl, control_type=torch.tensor([control_type_to_id("depth")], dtype=torch.long))
    assert not torch.allclose(out_a, out_b)


def test_encode_control_stack_supports_multi_control_inputs():
    enc = ControlNetEncoder(control_size=32, patch_size=2, in_channels=3, hidden_size=10, num_control_types=9)
    # (B=1, K=2, C=3, H=32, W=32)
    ctrl = torch.randn(1, 2, 3, 32, 32)
    out = encode_control_stack(
        enc,
        ctrl,
        target_hw=(32, 32),
        control_type=torch.tensor([control_type_to_id("canny"), control_type_to_id("depth")], dtype=torch.long),
        control_scale=torch.tensor([0.8, 0.6], dtype=torch.float32),
    )
    assert out.shape == (1, 256, 10)
