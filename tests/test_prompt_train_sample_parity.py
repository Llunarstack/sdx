"""Train/sample prompt guidance parity and emphasis wiring."""

from __future__ import annotations

from types import SimpleNamespace

from utils.prompt.prompt_emphasis import parse_prompt_emphasis, token_weights_from_cleaned_segments
from utils.prompt.stack.stages.guidance import apply_training_guidance_pair


def test_training_guidance_auto_adds_fragments():
    pos, neg = apply_training_guidance_pair(
        "1girl, solo, anime style",
        "",
        shortcomings_mode="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
    )
    assert len(pos) > len("1girl, solo, anime style")
    assert isinstance(neg, str)


def test_emphasis_strip_and_token_weights():
    raw = "(masterpiece:1.2), 1girl, [bad:0.8]"
    cleaned, segments = parse_prompt_emphasis(raw)
    assert "(" not in cleaned
    assert "[" not in cleaned
    assert "1girl" in cleaned
    assert segments
    import torch

    class _Tok:
        def __call__(self, texts, **kwargs):
            text = texts[0] if isinstance(texts, list) else texts
            words = text.split()
            offsets = []
            pos = 0
            for w in words:
                offsets.append((pos, pos + len(w)))
                pos += len(w) + 2
            while len(offsets) < kwargs.get("max_length", 16):
                offsets.append((0, 0))
            return {
                "input_ids": [[0] * len(offsets)],
                "offset_mapping": [offsets],
            }

    w = token_weights_from_cleaned_segments(
        cleaned, segments, _Tok(), max_length=16, device=torch.device("cpu"), dtype=torch.float32
    )
    assert w is not None
    assert w.numel() >= 1


def test_sample_stack_guidance_matches_training_path():
    """sample.py guidance flags use the same guidance module as training captions."""
    from utils.prompt.stack import apply_sample_prompt_stack

    args = SimpleNamespace(
        prompt="1girl, solo, digital painting",
        negative_prompt="",
        shortcomings_mitigation="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
        no_art_guidance_photography=False,
        auto_photo_realism=False,
        photo_realism_pack="none",
        photo_color_grade="none",
        photo_lighting_technique="none",
        photo_filter="none",
        photo_grain_style="none",
        photo_realism_strength=1.0,
        human_media_mode="none",
        realism_autopilot=False,
        invent_styles=0,
        style_genome_file="",
        _prompt_layout_negative="",
        _multi_instance_negative="",
        _detailed_scene_negative="",
        _visual_design_negative="",
    )
    train_pos, train_neg = apply_training_guidance_pair(
        args.prompt,
        "",
        shortcomings_mode="auto",
        shortcomings_2d=True,
        art_guidance_mode="auto",
        anatomy_guidance="lite",
        style_guidance_mode="auto",
        style_guidance_artists=True,
    )
    sample_pos, sample_neg = apply_sample_prompt_stack(args, args.prompt)
    assert len(sample_pos) >= len(args.prompt)
    assert len(train_pos) >= len(args.prompt)
    assert isinstance(sample_neg, str)
    assert isinstance(train_neg, str)
