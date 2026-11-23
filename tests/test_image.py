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
from utils.captioning import profile_image
from utils.generation.editing import parse_part_requests, visual_facts_from_requests
from utils.quality.finishing import fit_image_to_size
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
    from utils.generation.sample_run import sample_one_image_pil

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
    import utils.generation.sample_run as sm

    main = next(
        n
        for n in ast.parse(Path(sm.__file__).read_text(encoding="utf-8")).body
        if getattr(n, "name", "") == "_generate"  # every sample.py run, single or --jobs-jsonl, saves here
    )
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    guard = [c.lineno for c in calls if c.func.id == "require_image_allowed" and ast.unparse(c.args[0]) == "finals"]
    saves = [c.lineno for c in calls if c.func.id in ("_save", "save_rgb", "save_game_image")]
    assert guard and saves and max(guard) < min(saves)


def test_conditioning_image_paths_cover_every_image_flag() -> None:
    from utils.generation.sample_cli_parser import build_sample_parser
    from utils.generation.sample_core import conditioning_image_paths

    flags = {
        "--init-image": "init.png",
        "--edit-image": "edit.png",
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
    want = {"init", "edit", "ctl", "ref", "post", "rag", "rag1", "rag2", "dis", "style", "mood", "ctl2"}
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
        from utils.captioning import WDTag

        child = 0.3 if "kid" in Path(img).name else 0.05  # above SCREEN_THRESHOLD, below the caption threshold
        return [WDTag("solo", "general", 0.9), WDTag("female_child", "general", child)]

    def select(self, tags):
        return [t for t in tags if t.score >= self.threshold]


def test_wd_screen_drops_youth_rows(tmp_path, monkeypatch) -> None:
    import setup.manifests as tmw

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
    from utils.captioning import WDTagger

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
# test_image_dissection
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
    from utils.generation.editing import dissect_images_to_parts

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
# test_image_resize: Tests for utils.quality.finishing fitting modes.
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


# --- tiled denoising (MultiDiffusion) + outpainting --------------------------------------------------------


def _blur3(x, t, **_):
    return torch.nn.functional.avg_pool2d(torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)


@pytest.mark.parametrize("length,tile,overlap,align", [(56, 16, 4, 2), (40, 16, 8, 2), (33, 16, 4, 1), (16, 16, 4, 2)])
def test_tile_starts_cover_aligned(length, tile, overlap, align):
    from utils.generation.tiled import tile_starts

    starts = tile_starts(length, tile, overlap, align)
    assert starts[0] == 0 and starts[-1] == max(0, length - tile)
    assert all(s % align == 0 for s in starts[:-1])
    assert all(b - a <= tile - overlap + align for a, b in zip(starts, starts[1:], strict=False))


@pytest.mark.parametrize("h,w,tile,overlap,align", [(40, 56, 16, 8, 2), (32, 48, 16, 4, 2), (16, 40, 16, 4, 1)])
def test_tile_blend_weights_sum_to_one(h, w, tile, overlap, align):
    from utils.generation.tiled import tile_layout

    total = torch.zeros(h, w)
    for y, x, wt in tile_layout(h, w, tile, overlap, align):
        assert wt.min() > 0
        total[y : y + min(tile, h), x : x + min(tile, w)] += wt
    assert torch.allclose(total, torch.ones(h, w), atol=1e-6)


def test_tiled_identity_constant_and_kwargs():
    from utils.generation.tiled import TiledModel

    x, t = torch.randn(2, 4, 40, 56), torch.tensor([5, 5])
    assert torch.allclose(TiledModel(lambda x, t, **_: x, 16, 4, align=2)(x, t), x, atol=1e-6)
    const = TiledModel(lambda x, t, **_: torch.full((x.shape[0], 8, *x.shape[-2:]), 3.0), 16, 4, align=2)
    out = const(x, t)
    assert out.shape == (2, 8, 40, 56) and torch.allclose(out, torch.full_like(out, 3.0))

    def model(x, t, encoder_hidden_states, control_image, block_cache=None):
        assert block_cache is None and t.shape[0] == x.shape[0] == encoder_hidden_states.shape[0]
        assert control_image.shape[-2:] == (8 * x.shape[-2], 8 * x.shape[-1])
        pooled = torch.nn.functional.avg_pool2d(control_image, 8)[:, :1]
        return x + pooled + encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)

    kw = dict(encoder_hidden_states=torch.randn(2, 7, 5), control_image=torch.randn(2, 3, 320, 448))
    full = model(x, t, **kw)
    tiled = TiledModel(model, 16, 4, align=2, max_batch=3)(x, t, block_cache=object(), **kw)
    assert torch.allclose(tiled, full, atol=1e-5)


def test_tiled_local_model_close_to_full_frame():
    from utils.generation.tiled import TiledModel

    torch.manual_seed(0)
    x = torch.randn(2, 4, 40, 56)
    full = _blur3(x, None)
    blended = (TiledModel(_blur3, 16, 8, align=2)(x, torch.zeros(2)) - full).abs().max()
    hard = (TiledModel(_blur3, 16, 0, align=2)(x, torch.zeros(2)) - full).abs().max()
    assert blended < 0.03 and blended < 0.2 * hard


def test_tiled_prediction_continuous_across_tiles():
    from utils.generation.tiled import TiledModel

    ramp = torch.linspace(0, 1, 56).expand(1, 1, 40, 56).contiguous()
    tile_mean = TiledModel(lambda x, t, **_: x.mean(dim=(-2, -1), keepdim=True).expand_as(x), 16, 8, align=2)
    out = tile_mean(ramp, torch.zeros(1))
    jump = 8 / 55  # a hard cut steps by the difference of neighbouring tile means
    assert out.diff(dim=-1).abs().max() < 0.25 * jump
    assert out.diff(dim=-2).abs().max() < 1e-6


@pytest.mark.parametrize("flow", [False, True])
def test_tiled_model_inside_sample_loop_matches_full(flow):
    from diffusion.gaussian_diffusion import create_diffusion
    from utils.generation.tiled import TiledModel

    def model(x, t, encoder_hidden_states=None, **_):
        return 0.1 * x + encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)

    extra = dict(flow_matching_sample=True, flow_solver="euler") if flow else dict(solver="ddim", scheduler="ddim")
    kw = dict(
        model_kwargs_cond={"encoder_hidden_states": torch.ones(1, 3, 4)},
        model_kwargs_uncond={"encoder_hidden_states": torch.zeros(1, 3, 4)},
        cfg_scale=4.0,
        num_inference_steps=4,
        device="cpu",
        **extra,
    )
    outs = []
    for mdl in (model, TiledModel(model, 16, 4, align=2)):
        torch.manual_seed(0)
        outs.append(create_diffusion().sample_loop(mdl, (1, 4, 24, 40), **kw))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def test_maybe_tiled_auto_threshold_and_force():
    from utils.generation.tiled import TiledModel, maybe_tiled

    m = SimpleNamespace(latent_size=32, model=SimpleNamespace(patch_size=2))
    assert maybe_tiled(m, 32, 32, force=True) is None
    assert maybe_tiled(m, 32, 48) is None  # 1.5x native tokens: full attention is still fine
    assert isinstance(maybe_tiled(m, 32, 48, force=True), TiledModel)
    tm = maybe_tiled(m, 48, 64, overlap_px=60)
    assert isinstance(tm, TiledModel) and (tm.tile, tm.overlap, tm.align) == (32, 6, 2)
    assert maybe_tiled(m, 48, 64).overlap == 8


def test_hires_fix_passes_tiled_model():
    from utils.generation import sample_run as sm
    from utils.generation.tiled import TiledModel

    def run(width, **flags):
        args = SimpleNamespace(
            hires_fix=True, mask="", init_image="", init_latent="", width=width, height=width, **flags
        )
        m = SimpleNamespace(ae_type="kl", req_h=width, req_w=width, latent_size=32, model=SimpleNamespace(patch_size=2))
        s = SimpleNamespace(args=args, m=m, cfg_scale=5.0, renoise_refine=MagicMock(return_value="ok"))
        assert sm._hires_fix(s, torch.zeros(1, 4, 32, 32)) == "ok"
        return s.renoise_refine.call_args[0][-1]

    assert run(304) is None  # 38x38 latent is under 1.5x the native tokens
    assert isinstance(run(304, hires_tiled=True), TiledModel)
    assert isinstance(run(512), TiledModel)


def test_outpaint_flags_parse():
    from utils.generation.editing import parse_outpaint
    from utils.generation.sample_cli_parser import build_sample_parser

    args = build_sample_parser().parse_args(
        ["--ckpt", "x.pt", "--outpaint", "64,0,32,16", "--hires-tiled", "--tile-overlap", "48"]
    )
    assert (args.outpaint, args.hires_tiled, args.tile_overlap) == ("64,0,32,16", True, 48)
    assert parse_outpaint(args.outpaint) == (64, 0, 32, 16)
    for bad in ("1,2,3", "0,0,0,0", "-1,0,0,4", "a,b,c,d"):
        with pytest.raises(ValueError):
            parse_outpaint(bad)


def test_outpaint_canvas_geometry():
    from utils.generation.editing import outpaint_canvas

    canvas, mask = outpaint_canvas(Image.new("RGB", (100, 80), (200, 30, 40)), (40, 0, 20, 10), feather=12)
    assert canvas.size == mask.size == (160, 90)
    rgb, mk = np.asarray(canvas).astype(int), np.asarray(mask)
    assert (rgb[0:80, 40:140] == (200, 30, 40)).all()  # original kept
    assert np.abs(rgb[:, :40] - (200, 30, 40)).max() <= 2  # fill is the extended colour, not black
    assert (mk[:, :25] > 250).all() and (mk[:, 148:] > 250).all() and (mk[85:] > 250).all()
    assert (mk[0:45, 75:105] < 5).all()  # original interior stays known
    assert (mk[40, 40:50] >= 128).all() and (mk[40, 55:125] < 128).all()  # feather band into the original
    assert (mk[0, 60:120] < 128).all()  # top had no padding: its original edge stays known


def test_apply_outpaint_args(tmp_path):
    from utils.generation.editing import apply_outpaint_args

    src = tmp_path / "in.png"
    Image.new("RGB", (100, 80), (10, 20, 30)).save(src)
    base = dict(outpaint="40,0,20,10", mask="", init_latent="", width=0, height=0, inpaint_mode="legacy")
    args = SimpleNamespace(**base, init_image=str(src), strength=0.75, _explicit_flags=frozenset())
    apply_outpaint_args(args)
    assert (args.width, args.height, args.inpaint_mode, args.strength) == (160, 90, "mdm", 1.0)
    assert Image.open(args.init_image).size == Image.open(args.mask).size == (160, 90)
    typed = SimpleNamespace(**base, init_image=str(src), strength=0.6, _explicit_flags=frozenset({"strength"}))
    apply_outpaint_args(typed)
    assert typed.strength == 0.6
    with pytest.raises(SystemExit):
        apply_outpaint_args(SimpleNamespace(outpaint="8,8,8,8", init_image=str(src), mask="m.png", init_latent=""))


def test_outpaint_canvas_snaps_to_stride_without_area_cap():
    from diffusion.sampling import sampling_canvas_hw

    h, w = sampling_canvas_hw(490, 660, native_px=256, stride_px=16, max_area_ratio=float("inf"))
    assert (h % 16, w % 16) == (0, 0) and abs(h - 490) <= 8 and abs(w - 660) <= 8
