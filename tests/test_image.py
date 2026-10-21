"""Image layer of the minor guard (zero-shot CLIP age check) and the WD-tag manifest screen.

Also holds what used to be: test_image_age_guard, test_image_dissection, test_image_profiler, test_image_resize.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from utils.caption.image_profiler import profile_image
from utils.generation.image_dissection import parse_part_requests, visual_facts_from_requests
from utils.image_resize import fit_image_to_size
from utils.safety import minor_guard as mg
from utils.safety.minor_guard import ProhibitedContentError, ZeroShotAge, reasons, require_image_allowed

# ==================================================================================================
# test_image_age_guard: Image layer of the minor guard (zero-shot CLIP age check) and the WD-tag manifest screen.
# ==================================================================================================

_PROMPT_CLASS = {
    t.format(p): k for k, (_, phrases) in enumerate(mg._AGE_CLASSES) for p in phrases for t in mg._AGE_TEMPLATES
}


def _fake_text(texts):
    return torch.eye(4)[[_PROMPT_CLASS[t] for t in texts]]


def _fake_images(views):
    """Red reads as a child, green as an adult, blue or the gray padding as no person."""
    rows = [(np.asarray(v, np.float32).reshape(-1, 3).mean(0) - 127.0).clip(min=0.0) for v in views]
    return torch.tensor([[r, 0.0, g, b + 1e-3] for r, g, b in rows])


def _solid(rgb, h=64, w=64):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


RED, GREEN, GRAY = _solid((255, 0, 0)), _solid((0, 255, 0)), _solid((127, 127, 127))


@pytest.fixture
def fake_age(monkeypatch):
    monkeypatch.setattr(mg, "_age_model", lambda: ZeroShotAge(_fake_text, _fake_images))
    monkeypatch.setattr(mg, "_nsfw_scorer", lambda: None)


def _never(*_a, **_k):
    raise AssertionError("must not run")


def test_zero_shot_age_scores_whole_view_and_halves() -> None:
    mixed = np.concatenate([_solid((255, 0, 0)), _solid((0, 255, 0))], axis=1)  # child left, adult right
    (red, _), (green, _), gray, (mixed_whole, mixed_half) = ZeroShotAge(_fake_text, _fake_images).child_scores(
        [RED, GREEN, GRAY, mixed]
    )
    assert red > 0.99 and green < 0.01 and gray == (0.0, 0.0)
    assert mixed_half > 0.99 > mixed_whole  # the half view keeps a child beside an adult from averaging away


def test_sexual_image_of_child_is_refused(fake_age) -> None:
    with pytest.raises(ProhibitedContentError, match="minor"):
        require_image_allowed([GREEN, RED], sexual=True)
    require_image_allowed(GREEN, sexual=True)


def test_non_sexual_prompt_skips_the_age_check(monkeypatch) -> None:
    monkeypatch.setattr(mg, "_age_model", _never)
    monkeypatch.setattr(mg, "_nsfw_scorer", lambda: None)
    require_image_allowed(RED, sexual=False)
    monkeypatch.setattr(mg, "_nsfw_scorer", _never)
    require_image_allowed(["missing.png", RED], sexual=False, conditioning=True)


def test_nsfw_probability_above_half_triggers_the_age_check(fake_age, monkeypatch) -> None:
    monkeypatch.setattr(mg, "_nsfw_scorer", lambda: lambda im: 0.9)
    with pytest.raises(ProhibitedContentError):
        require_image_allowed(RED, sexual=False)
    monkeypatch.setattr(mg, "_nsfw_scorer", lambda: lambda im: 0.2)
    monkeypatch.setattr(mg, "_age_model", _never)
    require_image_allowed(RED, sexual=False)


def test_missing_weights_refuse_sexual_conditioning_but_warn_for_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mg, "_age_model", lambda: None)
    ref = tmp_path / "ref.png"
    Image.fromarray(GREEN).save(ref)
    with pytest.raises(ProhibitedContentError, match="weights"):
        require_image_allowed([str(ref)], sexual=True, conditioning=True)
    require_image_allowed(["", str(tmp_path / "missing.png")], sexual=True, conditioning=True)
    with pytest.warns(RuntimeWarning, match="age check skipped"):
        require_image_allowed(GREEN, sexual=True)


@pytest.mark.parametrize(("rgb", "refused"), [((1.0, -1.0, -1.0), True), ((-1.0, 1.0, -1.0), False)])
@patch("sample.encode_text", return_value=torch.zeros(2, 2, 8))
def test_flagged_library_output_is_never_returned(_enc, fake_age, rgb, refused) -> None:
    from utils.generation.simple_latent_generate import sample_one_image_pil

    vae = MagicMock()
    vae.decode.return_value = SimpleNamespace(sample=torch.tensor(rgb).view(1, 3, 1, 1).expand(1, 3, 32, 32))
    diffusion = MagicMock()
    diffusion.sample_loop.return_value = torch.zeros(1, 4, 4, 4)
    kw = dict(model=MagicMock(), diffusion=diffusion, tokenizer=None, text_encoder=MagicMock(), vae=vae, device="cpu")
    if refused:
        with pytest.raises(ProhibitedContentError):
            sample_one_image_pil(prompt="nude woman", image_size=32, **kw)
    else:
        assert sample_one_image_pil(prompt="nude woman", image_size=32, **kw).size == (32, 32)


def test_sample_main_screens_outputs_before_any_save() -> None:
    import utils.generation.sample_main as sm

    main = next(
        n
        for n in ast.parse(Path(sm.__file__).read_text(encoding="utf-8")).body
        if getattr(n, "name", "") == "_generate"  # every sample.py run, single or --jobs-jsonl, saves here
    )
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    guard = [c.lineno for c in calls if c.func.id == "require_image_allowed" and ast.unparse(c.args[0]) == "finals"]
    saves = [c.lineno for c in calls if c.func.id in ("_save", "save_rgb")]
    assert guard and saves and max(guard) < min(saves)


def test_conditioning_image_paths_cover_every_image_flag() -> None:
    from utils.generation.sample_cli_parser import build_sample_parser
    from utils.generation.sample_main import conditioning_image_paths

    flags = {
        "--init-image": "init.png",
        "--control-image": "ctl.png",
        "--reference-image": "ref.png",
        "--post-reference-image": "post.png",
        "--creative-rag-image": "rag.png",
        "--creative-rag-images": "rag1.png,rag2.png",
        "--dissect-refs": "dis.png",
        "--style-ref": "style.png:0.5",
        "--moodboard-images": "mood.png",
        "--control": "ctl2.png:canny:0.7",
    }
    args = build_sample_parser().parse_args(["--ckpt", "x.pt", "--prompt", "p", *sum(flags.items(), ())])
    want = {"init", "ctl", "ref", "post", "rag", "rag1", "rag2", "dis", "style", "mood", "ctl2"}
    assert {Path(p).stem for p in conditioning_image_paths(args)} == want


def test_youth_lexicon_edges() -> None:
    assert reasons("child bearing hips, nude woman") == ()
    assert reasons("child bearing a sword, nude")
    assert reasons("nude, randoseru")


class _FakeTagger:
    tag_names = ["1girl", "female_child", "solo"]

    def __init__(self, model_dir=None, *, threshold=0.35):
        self.model_dir, self.threshold = Path("wd"), threshold

    def scores(self, img):
        from utils.caption.wd_tagger import WDTag

        child = 0.3 if "kid" in Path(img).name else 0.05  # above SCREEN_THRESHOLD, below the caption threshold
        return [WDTag("solo", "general", 0.9), WDTag("female_child", "general", child)]

    def select(self, tags):
        return [t for t in tags if t.score >= self.threshold]


def test_wd_screen_drops_youth_rows(tmp_path, monkeypatch) -> None:
    import setup.tag_manifest_wd as tmw

    for name in ("adult.png", "kid.png", "tagged.png"):
        Image.fromarray(GREEN).save(tmp_path / name)
    rows = [
        {"image_path": "adult.png", "caption": "1girl, solo"},
        {"image_path": "kid.png", "caption": "1girl, solo"},
        {"image_path": "adult.png", "caption": "1girl, child"},
        {"image_path": "tagged.png", "caption": "1girl", "wd_tags": ["aged_down"], "tag_sources": ["wd_tagger"]},
        {"image_path": "gone.png", "caption": "landscape"},
    ]
    manifest, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(tmw, "WDTagger", _FakeTagger)
    assert tmw.tag_manifest(manifest, tmp_path, out) == 0
    kept = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["image_path"] for r in kept] == ["adult.png", "gone.png"]
    assert kept[0]["wd_tags"] == ["solo"]

    monkeypatch.setattr(_FakeTagger, "tag_names", ["1girl", "solo"])
    with pytest.raises(SystemExit, match="no youth tags"):
        tmw.tag_manifest(manifest, tmp_path, out)


def test_wd_tagger_input_format_and_categories(tmp_path, monkeypatch) -> None:
    from utils.caption.wd_tagger import WDTagger

    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n1,general,9,0\n2,long_hair,0,0\n3,hatsune_miku,4,0\n", encoding="utf-8"
    )
    seen = {}

    class Session:
        def __init__(self, *_a, **_k):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=["batch", 4, 4, 3])]

        def run(self, _out, feeds):
            seen["x"] = feeds["input"]
            return [np.array([[0.9, 0.5, 0.1]], dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    img = np.zeros((2, 4, 4), dtype=np.uint8)
    img[..., 0] = 255  # red, opaque
    img[..., 3] = 255
    Image.fromarray(img, "RGBA").save(tmp_path / "x.png")
    tagger = WDTagger(tmp_path, threshold=0.35)
    tags = tagger.scores(tmp_path / "x.png")
    assert [(t.name, t.category) for t in tags] == [
        ("general", "rating"),
        ("long_hair", "general"),
        ("hatsune_miku", "character"),
    ]
    assert [t.name for t in tagger.select(tags)] == ["long_hair"]  # no rating/character, no second sigmoid
    x = seen["x"]
    assert x.shape == (1, 4, 4, 3) and x.dtype == np.float32
    assert x[0, 0, 0].tolist() == [255.0, 255.0, 255.0]  # white padding row
    assert x[0, 1, 0].tolist() == [0.0, 0.0, 255.0]  # red in BGR order, unnormalized


# ==================================================================================================
# test_image_dissection: test_image_dissection
# ==================================================================================================


def test_parse_part_requests_simple() -> None:
    p = "use the hat from image 1 and take background from ref2"
    reqs = parse_part_requests(p)
    assert len(reqs) >= 2
    assert any(r.part.lower().strip() == "hat" and r.source_index == 0 for r in reqs)
    assert any(r.part.lower().strip() == "background" and r.source_index == 1 for r in reqs)


def test_visual_facts_mentions_parts() -> None:
    p = "use the sword from image 2"
    reqs = parse_part_requests(p)
    facts = visual_facts_from_requests(p, reqs, num_reference_images=2)
    joined = "\n".join(facts).lower()
    assert "sword" in joined
    assert "reference image 2" in joined


def test_dissect_background_full_mask(tmp_path: Path) -> None:
    # Heavy models are not required for background extraction.
    from utils.generation.image_dissection import dissect_images_to_parts

    img_p = tmp_path / "a.png"
    Image.new("RGB", (64, 32), (10, 20, 30)).save(img_p)

    reqs, parts, facts = dissect_images_to_parts(
        "take background from image 1",
        [str(img_p)],
        output_dir=tmp_path / "out",
        enable_heavy_models=True,  # still should work without model weights
    )
    assert reqs
    assert parts and parts[0].mask_path and parts[0].crop_path
    assert any("background" in f.lower() for f in facts)


# ==================================================================================================
# test_image_profiler: Tests for image profiler fusion logic.
# ==================================================================================================


def test_booru_metadata_wins(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    row = {
        "caption": "1girl, solo, long hair",
        "character_tags": ["hatsune_miku"],
        "copyright_tags": ["vocaloid"],
        "artist_tags": ["kantoku"],
    }
    prof = profile_image(img, booru_row=row, use_reverse_search=False, use_vlm=False)
    assert "hatsune miku" in prof.caption
    assert "vocaloid" in prof.caption
    assert "kantoku" in prof.caption
    assert prof.is_original_character is False
    assert prof.confidence >= 0.9
    assert prof.scene_summary
    assert "hatsune miku" in prof.scene_summary.lower()


def test_oc_when_no_identity(tmp_path):
    img = tmp_path / "y.png"
    img.write_bytes(b"fake")
    row = {"caption": "1girl, solo, blue hair, original"}
    prof = profile_image(img, booru_row=row, use_reverse_search=False, use_vlm=False)
    assert prof.is_original_character is True
    assert prof.scene_summary


def test_metadata_only_without_image_file():
    row = {
        "caption": "1girl, solo, outdoors",
        "character_tags": ["rem_(re:zero)"],
        "copyright_tags": ["re:zero_kara_hajimeru_isekai_seikatsu"],
    }
    prof = profile_image(Path("_missing.png"), booru_row=row, use_reverse_search=False, use_vlm=False)
    assert "rem" in prof.caption
    assert prof.scene_summary
    assert prof.confidence >= 0.9


# ==================================================================================================
# test_image_resize: Tests for utils.image_resize fitting modes.
# ==================================================================================================


def _toy_image(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # salient bright block on right side
    img[max(0, h // 4) : max(1, h // 4 + h // 3), max(0, w - w // 4) : w - 1, :] = 255
    return img


def test_fit_stretch_size():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 64, 64, mode="stretch")
    assert out.shape == (64, 64, 3)


def test_fit_center_crop_size():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 64, 64, mode="center_crop")
    assert out.shape == (64, 64, 3)


def test_fit_saliency_crop_prefers_bright_region():
    img = _toy_image(40, 80)
    out = fit_image_to_size(img, 40, 40, mode="saliency_crop")
    # Right-biased bright block should survive saliency crop better than center crop.
    ctr = fit_image_to_size(img, 40, 40, mode="center_crop")
    assert float(out.mean()) >= float(ctr.mean())
