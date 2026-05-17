"""Unit tests for sample.py subprocess wrappers (mocked)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from utils.generation.sample_edit_runner import (
    build_sample_command,
    resolve_repo_root,
    run_edit_with_pillow,
)


def test_resolve_repo_contains_sample_py():
    root = resolve_repo_root()
    assert (root / "sample.py").is_file()


def test_build_command_txt2img():
    cmd = build_sample_command(ckpt="m.pt", prompt="hi", out_path="o.png", width=256, height=256)
    assert "sample.py" in cmd[1] or cmd[1].endswith("sample.py")
    assert "--ckpt" in cmd and "m.pt" in cmd
    assert "--prompt" in cmd and "hi" in cmd
    assert "--width" in cmd and "256" in cmd


def test_build_command_inpaint_requires_init():
    with pytest.raises(ValueError, match="init"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            init_image_path=None,
            mask_image_path="mask.png",
        )


def test_build_command_rejects_invalid_strength():
    with pytest.raises(ValueError, match="strength"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            strength=1.5,
        )


def test_build_command_rejects_non_positive_width():
    with pytest.raises(ValueError, match="width"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            width=0,
            height=512,
        )


def test_extra_args_rejects_blank_string():
    with pytest.raises(ValueError, match="extra_args"):
        build_sample_command(
            ckpt="m.pt",
            prompt="hi",
            out_path="o.png",
            extra_args=["--foo", ""],
        )


@patch("utils.generation.sample_edit_runner.run_sample_inference")
def test_run_edit_with_pillow_resizes_and_calls(mock_run, tmp_path):
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    fake_out = tmp_path / "out.png"
    Image.new("RGB", (64, 64), (200, 0, 0)).save(fake_out)

    def fake_run_side_effect(*, out_path, **kw):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (10, 20, 30)).save(out_path)
        return subprocess.CompletedProcess(args=[], returncode=0)

    mock_run.side_effect = fake_run_side_effect

    base = Image.new("RGB", (128, 128), (255, 0, 0))
    mask = Image.new("L", (10, 10), 255)
    img = run_edit_with_pillow(
        ckpt="dummy.pt",
        prompt="paint a moon",
        negative_prompt="",
        base_image=base,
        mask_image=mask,
        width=512,
        height=512,
        seed=123,
        keep_temp_dir=False,
        extra_args=None,
    )
    assert isinstance(img, Image.Image)
    assert mock_run.called
    call_kw = mock_run.call_args.kwargs
    assert call_kw["mask_image_path"] is not None
    assert Path(call_kw["init_image_path"]).suffix == ".png"


@patch("utils.generation.sample_edit_runner.run_sample_inference")
def test_run_edit_img2img_no_mask(mock_run, tmp_path):
    def fake_run_side_effect(*, out_path, **kw):
        assert kw.get("mask_image_path") is None
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (5, 5, 5)).save(out_path)
        return subprocess.CompletedProcess(args=[], returncode=0)

    mock_run.side_effect = fake_run_side_effect

    out = run_edit_with_pillow(
        ckpt="x.pt",
        prompt="same scene",
        negative_prompt="blur",
        base_image=Image.new("RGB", (8, 8), (1, 2, 3)),
        mask_image=None,
        width=512,
        height=512,
    )
    assert out.size == (4, 4)
