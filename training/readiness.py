"""Training-readiness gate: prove the learning pipeline on the user's config before an expensive run.

``python -m scripts.tools readiness check <train.py args>`` runs the checks below and writes a READY / NOT READY
report to ``<results_dir>/readiness/``; ``train.py`` refuses runs longer than ``--readiness-min-steps`` without a
READY report for the same git commit, config hash and dataset fingerprint (``--skip-readiness-gate`` overrides).

Learning checks run the real objective, timestep sampling, caption dropout, EMA and checkpoint/resume code of
``train.py`` on a tiny copy of the configured DiT class (``real_model=True``: the full architecture) with tiny
locally built T5 / VAE stand-ins whose dims match the config, so nothing downloads.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import shlex
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

CRITICAL = ("dataset", "forward_backward", "finite", "overfit", "checkpoint_roundtrip", "resume", "sampling")
_TINY_MODEL = "Readiness-Tiny"
_REPO = Path(__file__).resolve().parents[1]
# Run length, logging and machine-local fields: changing them does not invalidate a readiness report.
_HASH_EXCLUDE = frozenset(
    {
        "results_dir",
        "resume",
        "max_steps",
        "passes",
        "epochs",
        "log_every",
        "ckpt_every",
        "save_every_steps",
        "val_every",
        "early_stopping_patience",
        "wandb_project",
        "tensorboard_dir",
        "log_images_every",
        "log_images_prompt",
        "dry_run",
        "save_run_manifest",
        "strict_warnings",
        "num_workers",
        "prefetch_factor",
        "auto_num_workers",
        "cuda_stream_prefetch",
        "live_dashboard",
        "torch_cpu_num_threads",
        "torch_cpu_num_interop_threads",
        "local_rank",
        "world_size",
    }
)
_log = logging.getLogger("readiness")


# --- identity of a run: commit + config + dataset ------------------------------------------------------------------


def config_hash(cfg) -> str:
    from utils.runtime import to_plain_dict

    d = {k: v for k, v in to_plain_dict(cfg).items() if k not in _HASH_EXCLUDE}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_info() -> dict:
    from training.loop import _safe_git_info

    return _safe_git_info(_REPO)


def report_path(cfg, cfg_hash: str, data_fp: str) -> Path:
    return Path(cfg.results_dir) / "readiness" / f"{cfg_hash}-{data_fp}.json"


def _count_rows(cfg) -> int:
    from data.audit import _SCAN_EXTS, _dataset_source, _scan_dirs

    src = _dataset_source(cfg)
    if src.is_file():
        with src.open("rb") as f:
            return sum(1 for line in f if line.strip())
    return sum(1 for d in _scan_dirs(src) for p in d.iterdir() if p.suffix.lower() in _SCAN_EXTS)


def estimate_total_steps(cfg, *, cap: int) -> int:
    """Steps ``train.py`` will run (same passes > max_steps > epochs rule); the dataset is only counted when needed."""
    if getattr(cfg, "dry_run", False):
        return 1
    max_steps, passes = int(cfg.max_steps or 0), int(cfg.passes or 0)
    if max_steps > 0 and (passes <= 0 or max_steps <= cap):
        return max_steps
    per_epoch = max(1, _count_rows(cfg) // max(1, int(cfg.global_batch_size)))
    steps = (passes if passes > 0 else int(cfg.epochs)) * per_epoch
    return min(steps, max_steps) if max_steps > 0 else steps


def enforce_readiness_gate(cfg, *, min_steps: int, skip: bool) -> None:
    """Return when the run is short, skipped, or covered by a READY report; otherwise exit with the command to run."""
    steps = estimate_total_steps(cfg, cap=min_steps)
    if steps <= min_steps:
        return
    if skip:
        bar = "!" * 100
        print(
            f"{bar}\n!! --skip-readiness-gate: starting a {steps}-step run WITHOUT a readiness report.\n"
            f"!! Nothing has proven this config learns, checkpoints, resumes or samples.\n{bar}",
            file=sys.stderr,
        )
        return
    from data.audit import dataset_fingerprint

    path = report_path(cfg, config_hash(cfg), dataset_fingerprint(cfg))
    commit = git_info()["commit"]
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rep = {}
    if rep.get("ready") and rep.get("git", {}).get("commit") == commit:
        print(f"Readiness: READY report {path} (commit {commit[:12]})")
        return
    why = "no report" if not rep else ("NOT READY" if not rep.get("ready") else "report is for another git commit")
    argv = [a for a in sys.argv[1:] if not a.startswith(("--readiness-min-steps", "--skip-readiness-gate"))]
    raise SystemExit(
        f"Refusing a {steps}-step run (> --readiness-min-steps {min_steps}): {why} at {path}.\n"
        f"Run:  python -m scripts.tools readiness check {shlex.join(argv)}\n"
        "or pass --skip-readiness-gate."
    )


# --- tiny rig: the training step of train.py on stand-in encoders --------------------------------------------------


@dataclass
class Rig:
    cfg: Any
    device: Any
    tokenizer: Any
    text_encoder: Any
    vae: Any
    diffusion: Any
    images: Any
    captions: list
    latents: Any
    notes: list
    max_len: int = 32

    def build(self, seed: int):
        """Model, EMA, optimizer and GradScaler exactly as a fresh ``train.py`` process seeds them."""
        import torch
        from config.train_config import get_dit_build_kwargs
        from models import DiT_models_text
        from utils.training.perf import create_adamw_optimizer

        from training.loop import setup_lora_training

        torch.manual_seed(seed)
        kw = get_dit_build_kwargs(self.cfg, class_dropout_prob=0.0)
        model = DiT_models_text[self.cfg.model_name](**kw).to(self.device)
        if getattr(self.cfg, "lora_train", False):
            setup_lora_training(model, self.cfg, 0, _log)
        ema = copy.deepcopy(model).requires_grad_(False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = create_adamw_optimizer(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, fused=False)
        return model, ema, opt, torch.amp.GradScaler("cuda", enabled=False)

    def encode(self, captions):
        from utils.modeling.encoders import encode_text

        return encode_text(captions, self.tokenizer, self.text_encoder, self.device, max_length=self.max_len)

    def loss(self, model, step: int, *, t=None, noise=None, drop: bool = True):
        from training.loop import (
            conditioning_kwargs,
            diffusion_step_loss,
            drop_captions,
            get_scheduled_caption_dropout,
            step_timesteps,
            training_noise,
        )

        cfg, lat = self.cfg, self.latents
        sched = getattr(cfg, "caption_dropout_schedule", None)
        p = get_scheduled_caption_dropout(step, sched) if sched else float(cfg.caption_dropout_prob or 0.0)
        caps = drop_captions(self.captions, p if drop else 0.0)[0]
        enc = self.encode(caps)
        if t is None:
            t = step_timesteps(cfg, self.diffusion.num_timesteps, lat.shape[0], self.device, global_step=step)
        kw = conditioning_kwargs(
            cfg,
            {},
            lat,
            self.device,
            encoder_hidden=enc,
            encoder_hidden_neg=None,
            style_embedding=None,
            token_weights=None,
        )
        return diffusion_step_loss(
            cfg,
            self.diffusion,
            model,
            lat,
            t,
            kw,
            None,
            training_noise(cfg, lat) if noise is None else noise,
            refinement_prob=float(cfg.refinement_prob or 0.0),
            refinement_max_t=int(cfg.refinement_max_t),
        )

    def step(self, model, ema, opt, step: int, total: int) -> tuple[float, float]:
        """One optimizer step (LR schedule, clipping, EMA as in train.py); returns ``(loss, grad_norm)``."""
        import torch

        from training.loop import ema_decay_at, get_lr_cosine, update_ema

        cfg = self.cfg
        for g in opt.param_groups:
            g["lr"] = get_lr_cosine(step, total, min(2, total), cfg.lr, cfg.min_lr)
        loss = self.loss(model, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"))
        opt.step()
        update_ema(ema, model, ema_decay_at(step + 1, cfg.ema_decay))
        return float(loss.detach()), float(gn)

    def eval_loss(self, model) -> float:
        """Loss on fixed timesteps and noise, without caption dropout (comparable before / after training)."""
        import torch

        from training.loop import fixed_seed

        model.eval()
        n, T, total = self.latents.shape[0], self.diffusion.num_timesteps, 0.0
        with torch.no_grad(), fixed_seed(1234):
            for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                t = torch.full((n,), int(frac * (T - 1)), device=self.device, dtype=torch.long)
                total += float(self.loss(model, 0, t=t, noise=torch.randn_like(self.latents), drop=False))
        model.train()
        return total / 5


def _vocab_tokenizer(captions: list, out: Path):
    import tokenizers
    import transformers
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = sorted({w for c in captions for w, _ in Whitespace().pre_tokenize_str(c)})
    vocab = {w: i for i, w in enumerate(["<pad>", "</s>", "<unk>", *words])}
    tok = tokenizers.Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_max_length=300
    ).save_pretrained(out)
    return len(vocab)


def _load_samples(cfg, n: int, notes: list):
    """``n`` (image, caption) pairs through the training loader at the tiny resolution, else synthetic ones."""
    import torch
    from data import collate_t2i

    from training.loop import build_train_dataset

    try:
        from data.audit import _dataset_source

        src = _dataset_source(cfg)
        root = cfg.data_path if cfg.manifest_jsonl else None
        ds = build_train_dataset(cfg, str(src), data_root=root, latent_dir=None, res_buckets=[])
        if len(ds) == 0:
            raise ValueError("empty dataset")
        batch = collate_t2i([ds[i % len(ds)] for i in range(n)])
        notes.append(f"overfit set: {n} rows of {src.name}")
        return batch["pixel_values"], batch["captions"]
    except Exception as e:
        notes.append(f"overfit set: synthetic ({type(e).__name__}: {e})")
        g = torch.Generator().manual_seed(0)
        s = int(cfg.image_size)
        imgs = torch.rand(n, 3, s, s, generator=g) * 2 - 1
        return imgs, [f"a photo of object {i} on a table" for i in range(n)]


def build_rig(cfg, work: Path, *, real_model: bool, n_samples: int, device: str = "cpu") -> Rig:
    import diffusers
    import torch
    import transformers
    from config.train_config import get_dit_build_kwargs
    from diffusion import create_diffusion
    from models import DiT_models_text
    from utils.modeling.encoders import apply_latent_spec

    from training.loop import encode_images_vae, get_t5_and_vae

    notes: list[str] = []
    tcfg = copy.deepcopy(cfg)
    tcfg.text_encoder = str(work / f"tiny_{Path(str(cfg.text_encoder)).name.replace('/', '_')}")
    tcfg.vae_model = str(work / "tiny_vae")
    tcfg.results_dir, tcfg.resume, tcfg.init_from = str(work / "results"), None, None
    tcfg.latent_cache_dir, tcfg.resolution_buckets, tcfg.use_compile = None, [], False
    for name, off in (("text_encoder_mode", "t5"), ("autoencoder_type", "kl")):
        if str(getattr(tcfg, name, off)) != off:
            notes.append(f"{name}={getattr(tcfg, name)} not exercised (tiny rig uses {off})")
            setattr(tcfg, name, off)
    if float(getattr(tcfg, "repa_weight", 0.0)) > 0:
        notes.append("REPA loss not exercised (needs its pretrained encoder)")
        tcfg.repa_weight = 0.0
    if not real_model:
        with torch.device("meta"):
            ref = DiT_models_text[cfg.model_name](**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
        p = int(getattr(ref, "patch_size", 2))
        DiT_models_text[_TINY_MODEL] = partial(type(ref), depth=2, hidden_size=64, num_heads=2, patch_size=p)
        tcfg.model_name, tcfg.image_size = _TINY_MODEL, 8 * p * 4
        notes.append(f"tiny {type(ref).__name__}: depth 2, width 64, {tcfg.image_size}px")
    kw = get_dit_build_kwargs(tcfg, class_dropout_prob=0.0)
    images, captions = _load_samples(tcfg, n_samples, notes)
    vocab = _vocab_tokenizer(captions, Path(tcfg.text_encoder))
    t5 = transformers.T5Config(
        vocab_size=vocab, d_model=kw["text_dim"], d_kv=16, d_ff=32, num_layers=1, num_heads=2, decoder_start_token_id=0
    )
    torch.manual_seed(0)
    transformers.T5EncoderModel(t5).save_pretrained(tcfg.text_encoder)
    ae = diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=int(kw["in_channels"]),
        norm_num_groups=4,
        sample_size=int(tcfg.image_size),
    )
    with torch.no_grad():  # like a trained VAE: near-deterministic posterior, scaling_factor -> unit-variance latents
        c = int(kw["in_channels"])
        ae.quant_conv.weight[c:].zero_()
        ae.quant_conv.bias[c:].fill_(-30.0)
        ae.register_to_config(scaling_factor=1.0 / float(ae.encode(images).latent_dist.mode().std()))
    ae.save_pretrained(tcfg.vae_model)
    dev = torch.device(device)
    tokenizer, text_encoder, vae = get_t5_and_vae(dev, tcfg)
    apply_latent_spec(tcfg, vae)
    diffusion = create_diffusion(
        timestep_respacing=tcfg.timestep_respacing,
        num_timesteps=tcfg.num_timesteps,
        beta_schedule=tcfg.beta_schedule,
        prediction_type=getattr(tcfg, "prediction_type", "epsilon"),
    )
    with torch.no_grad():
        latents = encode_images_vae(images.to(dev), vae, tcfg.latent_scale)
    return Rig(tcfg, dev, tokenizer, text_encoder, vae, diffusion, images, captions, latents, notes)


# --- checks --------------------------------------------------------------------------------------------------------


def check_architecture(cfg) -> tuple[str, dict]:
    """Image -> latent -> patch grid -> pos-embed, latent channels vs the VAE, text dim vs the encoder, and the
    ``--resume`` / ``--init-from`` checkpoint against the configured model (built on the meta device: no memory)."""
    import torch
    from config.train_config import get_dit_build_kwargs
    from models import DiT_models_text

    errs, info, unverified = [], {}, []
    kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    with torch.device("meta"):
        model = DiT_models_text[cfg.model_name](**kw)
    p = int(getattr(model, "patch_size", 2))
    stride = 8 * p
    for h, w in [(cfg.image_size, cfg.image_size), *(getattr(cfg, "resolution_buckets", None) or [])]:
        if int(h) % stride or int(w) % stride:
            errs.append(f"{h}x{w} is not divisible by VAE stride x patch = {stride}")
    grid = cfg.image_size // 8 // p
    info["latent"] = f"{kw['in_channels']}x{cfg.image_size // 8}x{cfg.image_size // 8}"
    info["patch_grid"] = f"{grid}x{grid} (patch {p})"
    n_patches = getattr(model, "num_patches", None)
    if n_patches is not None and n_patches != grid * grid:
        errs.append(f"model.num_patches={n_patches} != {grid}x{grid}")
    pe = getattr(model, "pos_embed", None)
    if pe is not None and pe.dim() == 3 and pe.shape[1] != grid * grid:
        errs.append(f"pos_embed covers {pe.shape[1]} tokens, latent grid has {grid * grid}")
    try:
        from transformers import AutoConfig

        tc = AutoConfig.from_pretrained(cfg.text_encoder, local_files_only=True)
        d = int(getattr(tc, "d_model", 0) or getattr(tc, "hidden_size", 0))
        info["text_dim"] = f"model {kw['text_dim']} / encoder {d}"
        if str(getattr(cfg, "text_encoder_mode", "t5")) == "t5" and d != kw["text_dim"]:
            errs.append(f"text encoder width {d} != DiT text_dim {kw['text_dim']} (inferred from the encoder name)")
    except Exception as e:
        unverified.append(f"text encoder config not local ({type(e).__name__})")
    if str(getattr(cfg, "autoencoder_type", "kl")) == "kl":
        try:
            from diffusers import AutoencoderKL

            vc = AutoencoderKL.load_config(
                cfg.vae_model, subfolder=getattr(cfg, "vae_subfolder", "") or None, local_files_only=True
            )
            ch = int(vc.get("latent_channels", 4))
            info["vae_latent_channels"] = ch
            # train.py writes the VAE's channel count onto cfg before building the DiT, so this must agree.
            if ch != kw["in_channels"]:
                info["note"] = f"train.py builds the DiT with {ch} channels (VAE), cfg says {kw['in_channels']}"
        except Exception as e:
            unverified.append(f"VAE config not local ({type(e).__name__})")
    ck = getattr(cfg, "resume", None) or getattr(cfg, "init_from", None)
    if ck:
        from utils.modeling.checkpoints import load_checkpoint_file

        state = load_checkpoint_file(ck)
        sd = state.get("model") or state.get("ema") or {}
        want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        have = {k: tuple(v.shape) for k, v in sd.items()}
        missing = sorted(set(want) - set(have) - ({"edit_type_embed"} if cfg.init_from else set()))
        bad = sorted(k for k in set(want) & set(have) if want[k] != have[k])
        unexpected = sorted(set(have) - set(want))
        info["checkpoint"] = f"{ck}: {len(have)} tensors"
        if missing or bad or unexpected:
            errs.append(f"checkpoint mismatch: missing={missing[:5]} shape={bad[:5]} unexpected={unexpected[:5]}")
    if unverified:
        info["unverified"] = unverified
    info["errors"] = errs
    return ("FAIL" if errs else "PASS"), info


def _close(a: dict, b: dict) -> float:
    """Largest absolute difference over the tensors of two state dicts."""
    import torch

    worst = 0.0
    for k, v in a.items():
        if isinstance(v, torch.Tensor):
            worst = max(worst, float((v.float() - b[k].float()).abs().max()) if v.numel() else 0.0)
    return worst


def _opt_tensors(opt) -> dict:
    return {f"{i}.{k}": v for i, s in opt.state_dict()["state"].items() for k, v in s.items() if hasattr(v, "shape")}


def learning_checks(rig: Rig, work: Path, *, overfit_steps: int, overfit_drop: float, lr: float) -> dict:
    """forward_backward, finite, overfit, checkpoint_roundtrip, resume, sampling on the rig."""
    import torch
    from utils.modeling.checkpoints import atomic_torch_save, load_checkpoint_file, set_rng_state

    from training.loop import load_dit_weights, restore_trainer_state, trainer_state

    out: dict[str, tuple[str, dict]] = {}
    seed = int(rig.cfg.global_seed)
    model, ema, opt, _ = rig.build(seed)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    loss = rig.loss(model, 0)
    loss.backward()
    bad = [n for n, p in trainable.items() if p.grad is not None and not torch.isfinite(p.grad).all()]
    ok = bool(torch.isfinite(loss)) and not bad
    out["forward_backward"] = ("PASS" if ok else "FAIL", {"loss": float(loss.detach()), "nonfinite_grads": bad[:10]})
    opt.zero_grad(set_to_none=True)

    rig.cfg.lr = lr
    reached = dict.fromkeys(trainable, 0.0)
    before = rig.eval_loss(model)
    losses, norms, t0 = [], [], time.perf_counter()
    for s in range(overfit_steps):
        loss_v, gn = rig.step(model, ema, opt, s, overfit_steps)
        losses.append(loss_v)
        norms.append(gn)
        if s < 4:  # adaLN-Zero / layer-scale gates open after the first steps; by then every param needs a grad
            for n, p in trainable.items():
                reached[n] += float(p.grad.abs().sum()) if p.grad is not None else 0.0
    after = rig.eval_loss(model)
    dead = [n for n, v in reached.items() if v == 0.0]
    if dead:
        out["forward_backward"] = ("FAIL", {**out["forward_backward"][1], "params_without_grad": dead[:20]})
    finite = all(map(math.isfinite, losses + norms)) and all(torch.isfinite(p).all() for p in model.parameters())
    out["finite"] = ("PASS" if finite else "FAIL", {"steps": overfit_steps, "max_grad_norm_seen": max(norms)})
    target = before * (1 - overfit_drop)
    out["overfit"] = (
        "PASS" if after <= target else "FAIL",
        {
            "eval_loss_before": round(before, 5),
            "eval_loss_after": round(after, 5),
            "needed": f"<= {target:.5f} ({overfit_drop:.0%} drop)",
            "train_loss_first_last": [round(losses[0], 5), round(sum(losses[-10:]) / len(losses[-10:]), 5)],
            "seconds": round(time.perf_counter() - t0, 1),
        },
    )

    # Resume: K steps straight vs K/2, checkpoint, a fresh process (re-seeded build), load, K/2 more.
    k = 6

    def ckpt_of(m, e, o, sc, step):
        return {
            "model": m.state_dict(),
            "ema": e.state_dict(),
            "config": rig.cfg,
            **trainer_state(o, sc, steps=step, steps_per_epoch=1),
        }

    a_model, a_ema, a_opt, _ = rig.build(seed)
    for s in range(k):
        rig.step(a_model, a_ema, a_opt, s, k)
    b_model, b_ema, b_opt, b_sc = rig.build(seed)
    for s in range(k // 2):
        rig.step(b_model, b_ema, b_opt, s, k)
    path = work / "readiness_ckpt.pt"
    atomic_torch_save(ckpt_of(b_model, b_ema, b_opt, b_sc, k // 2), path)

    c = load_checkpoint_file(path)
    r_model, r_ema, r_opt, r_sc = rig.build(seed)
    load_dit_weights(r_model, c["model"], allow_new_edit_embed=False)
    load_dit_weights(r_ema, c["ema"], allow_new_edit_embed=False)
    restore_trainer_state(c, r_opt, r_sc, _log)
    diffs = {
        "model": _close(b_model.state_dict(), r_model.state_dict()),
        "ema": _close(b_ema.state_dict(), r_ema.state_dict()),
        "optimizer": _close(_opt_tensors(b_opt), _opt_tensors(r_opt)),
    }
    out["checkpoint_roundtrip"] = (
        "PASS" if not any(diffs.values()) else "FAIL",
        {"max_abs_diff": diffs, "keys": sorted(k_ for k_ in c if k_ != "config")},
    )
    if c.get("rng"):
        set_rng_state(c["rng"])
    for s in range(int(c["steps"]), k):
        rig.step(r_model, r_ema, r_opt, s, k)
    d_model = _close(a_model.state_dict(), r_model.state_dict())
    d_ema = _close(a_ema.state_dict(), r_ema.state_dict())
    exact = d_model <= 1e-6 and d_ema <= 1e-6
    out["resume"] = (
        "PASS" if exact else "FAIL",
        {
            "steps": f"{k} straight vs {k // 2} + resume + {k - k // 2}",
            "max_abs_diff_model": d_model,
            "max_abs_diff_ema": d_ema,
            "rng_in_checkpoint": bool(c.get("rng")),
        },
    )

    # Sampling: sample.py's entry point on the checkpoint written above.
    from PIL import Image
    from utils.generation.sample_run import main as sample_main

    img_path = work / "readiness_sample.png"
    size = str(int(rig.cfg.image_size))
    sample_main(
        [
            "--ckpt",
            str(path),
            "--prompt",
            rig.captions[0] or "a photo",
            "--out",
            str(img_path),
            "--device",
            str(rig.device),
        ]
        + ["--width", size, "--height", size, "--steps", "2", "--num", "1", "--seed", "1", "--no-quality-defaults"]
    )
    with Image.open(img_path) as im:
        im.load()
        good = im.size == (int(size), int(size))
        out["sampling"] = ("PASS" if good else "FAIL", {"size": list(im.size), "steps": 2})
    return out


def check_throughput(cfg, steps: int = 5) -> tuple[str, dict]:
    """Images/sec and peak VRAM of the configured model at the configured size and per-device batch (GPU only)."""
    import torch
    from config.train_config import get_dit_build_kwargs
    from diffusion import create_diffusion
    from models import DiT_models_text

    from training.loop import diffusion_step_loss, step_timesteps

    dev = torch.device("cuda")
    kw = get_dit_build_kwargs(cfg, class_dropout_prob=0.0)
    model = DiT_models_text[cfg.model_name](**kw).to(dev)
    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    diff = create_diffusion(num_timesteps=cfg.num_timesteps, beta_schedule=cfg.beta_schedule)
    b, s = int(cfg.per_device_batch_size), int(cfg.image_size) // 8
    lat = torch.randn(b, kw["in_channels"], s, s, device=dev)
    enc = torch.randn(b, 77, kw["text_dim"], device=dev)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
            t = step_timesteps(cfg, diff.num_timesteps, b, dev)
            loss = diffusion_step_loss(
                cfg,
                diff,
                model,
                lat,
                t,
                {"encoder_hidden_states": enc},
                None,
                torch.randn_like(lat),
                refinement_prob=0.0,
                refinement_max_t=0,
            )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)
    ips = b * len(times) / sum(times)
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    torch.cuda.empty_cache()
    return "PASS", {"images_per_sec": round(ips, 2), "peak_vram_gb": round(peak, 2), "batch": b, "dit_only": True}


def _run(name: str, fn, results: list, critical: bool) -> Any:
    t0 = time.perf_counter()
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "FAIL", {"error": f"{type(e).__name__}: {e}", "where": traceback.format_exc(limit=-3)}
    results.append(
        {
            "name": name,
            "status": status,
            "critical": critical,
            "seconds": round(time.perf_counter() - t0, 1),
            "detail": detail,
        }
    )
    return detail


def run_readiness(
    cfg,
    *,
    device: str = "auto",
    audit_limit: int = 5000,
    max_bad_frac: float = 0.01,
    overfit_steps: int = 400,
    overfit_samples: int = 8,
    overfit_drop: float = 0.15,
    overfit_lr: float = 0.0,
    real_model: bool = False,
    argv: list | None = None,
) -> dict:
    """Run every check; ``report["ready"]`` is True when all critical checks PASS. The report is also written to
    ``<results_dir>/readiness/<config hash>-<dataset fingerprint>.json``."""
    import torch
    from data.audit import audit_dataset
    from toolkit.env_health import collect_env

    cfg_hash = config_hash(cfg)
    cfg = copy.deepcopy(cfg)
    results: list[dict] = []
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    _run("env", lambda: ("PASS" if collect_env().get("torch") else "FAIL", collect_env()), results, False)
    audit = _run(
        "dataset", lambda: _audit(audit_dataset(cfg, limit=audit_limit, max_bad_frac=max_bad_frac)), results, True
    )
    _run("architecture", lambda: check_architecture(cfg), results, False)

    from models import DiT_models_text

    with tempfile.TemporaryDirectory(prefix="sdx_readiness_") as tmp:
        work = Path(tmp)
        rig_notes: list = []

        def _learn():
            rig = build_rig(cfg, work, real_model=real_model, n_samples=overfit_samples)
            rig_notes.extend(rig.notes)
            lr = overfit_lr or (float(cfg.lr) if real_model else 1e-3)
            return learning_checks(rig, work, overfit_steps=overfit_steps, overfit_drop=overfit_drop, lr=lr)

        learned: dict = {}
        _run("learning_rig", lambda: ("PASS", learned.update(_learn()) or {"notes": rig_notes}), results, False)
        DiT_models_text.pop(_TINY_MODEL, None)
        for name in ("forward_backward", "finite", "overfit", "checkpoint_roundtrip", "resume", "sampling"):
            st, det = learned.get(name, ("FAIL", {"error": "learning rig failed (see learning_rig)"}))
            results.append({"name": name, "status": st, "critical": True, "seconds": 0.0, "detail": det})
    if use_cuda:
        _run("throughput", lambda: check_throughput(cfg), results, False)
    else:
        results.append(
            {
                "name": "throughput",
                "status": "SKIP",
                "critical": False,
                "seconds": 0.0,
                "detail": {"reason": "no CUDA device (or --device cpu)"},
            }
        )

    ready = all(r["status"] == "PASS" for r in results if r["critical"])
    report = {
        "ready": ready,
        "verdict": "READY" if ready else "NOT READY",
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "config_hash": cfg_hash,
        "dataset_fingerprint": (audit or {}).get("fingerprint", ""),
        "argv": argv or [],
        "checks": results,
        "dataset_audit": (audit or {}).get("report"),
    }
    if report["dataset_fingerprint"]:
        path = report_path(cfg, cfg_hash, report["dataset_fingerprint"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["path"] = str(path)
    return report


def _audit(rep: dict) -> tuple[str, dict]:
    return ("PASS" if rep["ok"] else "FAIL"), {
        "fingerprint": rep["fingerprint"],
        "report": rep,
        "bad_frac": rep["bad_frac"],
    }


def _brief(detail: Any) -> str:
    if not isinstance(detail, dict):
        return str(detail)
    d = {k: v for k, v in detail.items() if k not in ("report", "where", "optional_libs", "sdx_native")}
    return json.dumps(d, default=str)[:160]


def format_report(rep: dict) -> str:
    rows = [f"{'check':<22}{'status':<7}{'crit':<6}detail", "-" * 100]
    for r in rep["checks"]:
        rows.append(f"{r['name']:<22}{r['status']:<7}{'yes' if r['critical'] else '':<6}{_brief(r['detail'])}")
    rows.append("-" * 100)
    rows.append(
        f"{rep['verdict']}  (commit {rep['git']['commit'][:12]}{' dirty' if rep['git'].get('dirty') else ''}, "
        f"config {rep['config_hash']}, dataset {rep['dataset_fingerprint']})"
    )
    if rep.get("path"):
        rows.append(f"report: {rep['path']}")
    return "\n".join(rows)
