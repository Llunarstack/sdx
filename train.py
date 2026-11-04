"""
Train the text-conditioned DiT (VP diffusion, rectified flow or masked MDM), on one GPU or with DDP.

``python train.py --help`` lists every option. ``--profile-out PATH`` (with ``--profile-sort`` /
``--profile-top``) writes cProfile output plus a sorted text summary, like sample.py.
"""

import os
import pickle
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from time import time

from utils.runtime import configure_stdio_for_console

configure_stdio_for_console()

# ``--help`` exits before ``import torch`` (tests/test_cli_entrypoints.py): ``training.cli`` imports only argparse.
if __name__ == "__main__" and any(_h in sys.argv[1:] for _h in ("-h", "--help")):
    from training.cli import build_train_arg_parser

    build_train_arg_parser().parse_args()

import torch
import torch.distributed as dist
from config.train_config import TrainConfig, get_dit_build_kwargs
from data import ResolutionBucketBatchSampler, collate_t2i
from diffusion import create_diffusion
from models import DiT_models_text
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from training.cli import apply_book_train_preset_to_args, build_train_arg_parser, build_train_config_from_args
from training.loop import (
    apply_runtime_ar,
    bridge_aux_loss,
    build_rae_bridge,
    build_train_dataset,
    check_latent_cache_spec,
    conditioning_kwargs,
    create_logger,
    diffusion_step_loss,
    edit_source_latents,
    ema_decay_at,
    ema_update_state,
    encode_images_vae,
    encode_kw_for_batch,
    eval_val_loss,
    get_curriculum_max_length,
    get_lr_cosine,
    get_scheduled_caption_dropout,
    get_t5_and_vae,
    load_dit_weights,
    log_sample_image,
    negatives_strip_emphasis,
    repa_alignment_loss,
    repa_features,
    repeat_epochs,
    seed_dataloader_worker,
    set_tf32,
    setup_lora_training,
    step_timesteps,
    training_noise,
    update_ema,
    write_run_manifest,
)
from utils.modeling.encoders import encode_kwargs_for_captions, encode_text, load_text_encoder_bundle
from utils.runtime import to_plain_dict
from utils.training.monitoring import estimate_memory_usage, validate_train_config
from utils.training.monitoring import (
    get_model_info,
    log_gpu_memory,
    log_system_info,
    print_model_summary,
    setup_logging,
)
from utils.training.objectives import parse_ar_order_mix, resolve_ar_for_step
from utils.training.perf import (
    configure_training_cuda_and_cpu,
    create_adamw_optimizer,
    dataloader_perf_kwargs,
    encode_neg_and_style_embeddings,
    log_cuda_quick_stats,
    log_training_perf_hints,
    maybe_cuda_prefetch_dataloader,
    resolve_training_num_workers,
    to_train_device,
)


def main(cfg: TrainConfig):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training requires a CUDA-capable GPU. Run `nvidia-smi` to check driver status."
        )

    logger = setup_logging()

    if bool(getattr(cfg, "strict_warnings", False)):
        warnings.filterwarnings(
            "error", category=UserWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)"
        )
        warnings.filterwarnings(
            "error", category=FutureWarning, module=r"^(train|config|data|diffusion|models|training|utils)(\.|$)"
        )
        logger.info("Strict warnings enabled: project UserWarning/FutureWarning are treated as errors.")

    logger.info("Validating configuration...")
    for issue in validate_train_config(cfg):
        if issue.startswith("ERROR"):
            logger.error(issue)
            raise ValueError(f"Configuration error: {issue}")
        logger.warning(issue)

    logger.info(f"System info: {log_system_info()}")
    memory_est = estimate_memory_usage(cfg)
    logger.info(f"Estimated memory usage: {memory_est['total_estimated_gb']:.1f}GB")

    use_ddp = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda", 0)
        world_size = 1

    cfg.world_size = world_size
    cfg.local_rank = local_rank
    torch.manual_seed(cfg.global_seed + rank)
    if getattr(cfg, "deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    set_tf32(bool(getattr(cfg, "enable_tf32", True)))
    configure_training_cuda_and_cpu(
        deterministic=bool(getattr(cfg, "deterministic", False)),
        cudnn_benchmark=bool(getattr(cfg, "cudnn_benchmark", True)),
        enable_tf32=bool(getattr(cfg, "enable_tf32", True)),
        cpu_num_threads=int(getattr(cfg, "torch_cpu_num_threads", 0) or 0),
        cpu_num_interop_threads=int(getattr(cfg, "torch_cpu_num_interop_threads", 0) or 0),
        logger=logger,
    )
    if rank == 0:
        log_cuda_quick_stats(logger)
        log_training_perf_hints(logger, int(cfg.num_workers), int(cfg.world_size))

    if rank == 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(cfg.results_dir) / f"{stamp}-{cfg.model_name.replace('/', '-')}"
        ckpt_dir = exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(str(exp_dir))
    else:
        exp_dir = Path(cfg.results_dir) / "run"
        logger = create_logger(None)

    if rank == 0 and bool(getattr(cfg, "save_run_manifest", True)):
        try:
            write_run_manifest(exp_dir, cfg, logger)
        except Exception as e:
            logger.warning(f"Failed to write run manifest: {e}")

    log_gpu_memory(logger, "Before model loading: ")
    logger.info("Loading T5 and autoencoder...")
    tokenizer, text_encoder, vae = get_t5_and_vae(device, cfg)
    text_bundle = load_text_encoder_bundle(cfg, device)
    if text_bundle is not None:
        if text_bundle.mode == "penta":
            logger.info(
                "Penta text encoder enabled: T5 + CLIP-L + CLIP-bigG + CLIP-H + LongCLIP-L "
                "(fusion layers trained with DiT)."
            )
        else:
            logger.info("Triple text encoder enabled: T5 + CLIP-L + CLIP-bigG (fusion layers trained with DiT).")

    ae_type = getattr(cfg, "autoencoder_type", "kl")
    if ae_type == "kl":
        from utils.modeling.encoders import apply_latent_spec

        spec = apply_latent_spec(cfg, vae)  # saved with the checkpoint, so sampling decodes identically
        logger.info(f"VAE {cfg.vae_model}: {spec.channels}-channel latents, scale={spec.scale:g}, shift={spec.shift:g}")
        check_latent_cache_spec(getattr(cfg, "latent_cache_dir", None), spec)
    effective_latent_scale = getattr(cfg, "latent_scale", 0.18215) if ae_type == "kl" else 1.0

    rae_bridge = build_rae_bridge(cfg, vae, device, logger) if ae_type == "rae" else None

    model_fn = DiT_models_text.get(cfg.model_name)
    if model_fn is None:
        raise ValueError(f"Unknown model {cfg.model_name}. Choices: {list(DiT_models_text.keys())}")
    # Caption dropout happens in the loop (captions -> "" before encoding), never inside the model.
    model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0)).to(device)

    if rank == 0:
        logger.info(f"Model info: {get_model_info(model)}")
        print_model_summary(model)

    if cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()

    lora_training = bool(getattr(cfg, "lora_train", False))
    if lora_training:
        setup_lora_training(model, cfg, rank, logger)

    def _maybe_save_lora_adapter(raw_model, ckpt_path) -> None:
        """Write an adapter-only file next to a full checkpoint (LoRA training)."""
        if not lora_training:
            return
        from models.lora import lora_state_dict

        adapter_path = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_lora.pt")
        torch.save(lora_state_dict(raw_model), adapter_path)
        logger.info(f"LoRA adapter saved: {adapter_path}")

    raw_model = model
    ema = deepcopy(model).requires_grad_(False)
    # The fusion layers and the RAE bridge are trained with the DiT and run next to its EMA weights at inference,
    # so they are averaged with the same decay.
    ema_bridge = deepcopy(rae_bridge).requires_grad_(False) if rae_bridge is not None else None
    fusion = text_bundle.fusion if text_bundle is not None else None
    ema_fusion = deepcopy(fusion).requires_grad_(False) if fusion is not None else None
    ema_pairs = [(m, e) for m, e in ((raw_model, ema), (rae_bridge, ema_bridge), (fusion, ema_fusion)) if m is not None]

    if use_ddp:
        # Edit models skip edit_type_embed on text-to-image steps, which plain DDP treats as an error.
        model = DDP(
            model, device_ids=[local_rank], find_unused_parameters=bool(getattr(cfg, "edit_conditioning", False))
        )

    if rank == 0:
        logger.info(f"Model: {cfg.model_name}")

    diffusion = create_diffusion(
        timestep_respacing=cfg.timestep_respacing,
        num_timesteps=cfg.num_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )

    # Only optimize params that need grad (LoRA training freezes the base model;
    # full training leaves everything trainable, so this is a no-op there).
    opt_params = [p for p in model.parameters() if p.requires_grad]
    if rae_bridge is not None:
        opt_params += list(rae_bridge.parameters())
    if text_bundle is not None and text_bundle.fusion is not None:
        opt_params += list(text_bundle.fusion.parameters())
    opt = create_adamw_optimizer(
        opt_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=bool(getattr(cfg, "fused_adamw", True)),
    )
    # Loss scaling only matters for FP16 underflow; BF16 has FP32's exponent range.
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(cfg, "use_fp16", False))

    start_step = 0
    best_loss = float("inf")

    def _load_ckpt_file(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:  # checkpoints pickle the TrainConfig
            return torch.load(path, map_location="cpu", weights_only=False)

    def _apply_train_checkpoint(ckpt: dict, *, load_optimizer: bool) -> None:
        new_edit = not load_optimizer  # --init-from may add edit conditioning; --resume must match exactly
        if load_dit_weights(raw_model, ckpt.get("model", ckpt.get("ema")), allow_new_edit_embed=new_edit):
            logger.info("Text-to-image checkpoint -> edit model: edit_type_embed initialised fresh.")
        load_dit_weights(ema, ckpt.get("ema") or raw_model.state_dict(), allow_new_edit_embed=new_edit)
        if load_optimizer and ckpt.get("opt"):
            try:
                opt.load_state_dict(ckpt["opt"])
            except (ValueError, KeyError) as e:
                logger.warning(f"Optimizer state not restored (fresh optimizer): {e}")
        for key, live, shadow in (
            ("rae_latent_bridge", rae_bridge, ema_bridge),
            ("text_encoder_fusion", fusion, ema_fusion),
        ):
            if live is None or not ckpt.get(key):
                continue
            try:
                live.load_state_dict(ckpt[key], strict=True)
                shadow.load_state_dict(ckpt.get(f"ema_{key}") or ckpt[key], strict=True)
                logger.info(f"Loaded {key} from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not load {key}: {e}")

    def _checkpoint_dict(step: int, **extra) -> dict:
        ckpt = {
            "model": raw_model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "steps": step,
            "config": cfg,
            **extra,
        }
        if rae_bridge is not None:
            ckpt["rae_latent_bridge"] = rae_bridge.state_dict()
            ckpt["ema_rae_latent_bridge"] = ema_bridge.state_dict()
        if fusion is not None:
            ckpt["text_encoder_fusion"] = fusion.state_dict()
            ckpt["ema_text_encoder_fusion"] = ema_fusion.state_dict()
        return ckpt

    if getattr(cfg, "resume", None):
        resume_path = Path(cfg.resume)
        if resume_path.exists():
            ckpt = _load_ckpt_file(resume_path)
            _apply_train_checkpoint(ckpt, load_optimizer=True)
            start_step = ckpt.get("steps", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            logger.info(f"Resumed from {resume_path} at step {start_step}")
    elif getattr(cfg, "init_from", None):
        init_path = Path(cfg.init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"--init-from is not a file: {init_path}")
        ckpt = _load_ckpt_file(init_path)
        _apply_train_checkpoint(ckpt, load_optimizer=False)
        logger.info(f"Initialized weights from {init_path} (step=0, fresh optimizer)")

    # Manifest JSONL rows resolve relative image paths against --data-path.
    manifest_path = cfg.manifest_jsonl
    data_root = cfg.data_path
    dataset_path = manifest_path or data_root
    if not dataset_path:
        raise ValueError("Set --data-path or --manifest-jsonl")

    if getattr(cfg, "dry_run", False):
        cfg.max_steps = 1
        logger.info("Dry run: 1 step then exit.")

    val_split = getattr(cfg, "val_split", 0.0)
    res_buckets = getattr(cfg, "resolution_buckets", None) or []
    latent_dir = getattr(cfg, "latent_cache_dir", None)
    if res_buckets:
        if use_ddp:
            raise ValueError(
                "--resolution-buckets is not supported with multi-GPU DDP yet. "
                "Use a single GPU or omit --resolution-buckets."
            )
        if val_split > 0 and val_split < 1:
            raise ValueError("--resolution-buckets requires --val-split 0 (no train/val split) in this version.")
        stride = 8 * int(getattr(raw_model, "patch_size", 2) or 2)
        for h, w in res_buckets:
            if int(h) % stride != 0 or int(w) % stride != 0:
                raise ValueError(
                    f"Resolution bucket ({h},{w}) must be divisible by {stride} (VAE stride x patch size)."
                )
        if latent_dir:
            logger.warning("Disabling latent cache (--latent-cache-dir) when using resolution buckets.")
            latent_dir = None

    dataset = build_train_dataset(
        cfg,
        dataset_path,
        data_root=data_root if manifest_path else None,
        latent_dir=latent_dir,
        res_buckets=res_buckets,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No training samples found. manifest={manifest_path!r} data_root={data_root!r}. "
            "Expected a folder of images with matching .txt/.caption sidecars "
            "(flat or one level of subfolders), or a manifest .jsonl with "
            "{'image_path', 'caption'} rows resolved against --data-path."
        )

    val_loader = None
    train_dataset = dataset
    train_num_workers = resolve_training_num_workers(
        int(cfg.num_workers),
        len(dataset),
        int(cfg.per_device_batch_size),
        auto=bool(getattr(cfg, "auto_num_workers", True)),
    )
    dl_common = dataloader_perf_kwargs(
        num_workers=train_num_workers,
        prefetch_factor=int(getattr(cfg, "prefetch_factor", 4)),
        worker_init_fn=partial(seed_dataloader_worker, base_seed=cfg.global_seed + rank * 1000)
        if getattr(cfg, "deterministic", False)
        else None,
    )
    cuda_prefetch = bool(getattr(cfg, "cuda_stream_prefetch", True))
    if rank == 0 and train_num_workers != int(cfg.num_workers):
        logger.info("DataLoader num_workers=%d (requested %d)", train_num_workers, int(cfg.num_workers))
    if rank == 0 and cuda_prefetch and device.type == "cuda":
        logger.info("CUDA stream prefetch enabled (overlap H2D with training step)")
    if rank == 0 and not getattr(cfg, "latent_cache_dir", None):
        logger.info(
            "Tip: precompute VAE latents (python -m scripts.tools precompute_latents) "
            "and pass --latent-cache-dir for 3-10x faster epochs when IO-bound."
        )
    if val_split > 0 and val_split < 1:
        n_total = len(dataset)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.global_seed)
        )
        if rank == 0:
            val_loader = maybe_cuda_prefetch_dataloader(
                DataLoader(
                    val_dataset,
                    batch_size=cfg.per_device_batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_t2i,
                    **dl_common,
                ),
                device,
                enabled=cuda_prefetch,
            )
        logger.info(f"Train/val split: {len(train_dataset)} train, {len(val_dataset)} val")
    else:
        val_dataset = None

    sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.global_seed)
        if use_ddp
        else None
    )
    if res_buckets:
        bucket_sampler = ResolutionBucketBatchSampler(
            train_dataset,
            cfg.per_device_batch_size,
            drop_last=True,
            shuffle_batches=True,
            generator=torch.Generator().manual_seed(cfg.global_seed),
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=collate_t2i,
            **dl_common,
        )
    else:
        loader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_t2i,
            generator=torch.Generator().manual_seed(cfg.global_seed + rank)
            if getattr(cfg, "deterministic", False)
            else None,
            **dl_common,
        )
    loader = maybe_cuda_prefetch_dataloader(loader, device, enabled=cuda_prefetch)
    logger.info(f"Train size: {len(train_dataset)}, batch per device: {cfg.per_device_batch_size}")

    # Training length: passes (best) > max_steps > epochs
    max_steps = getattr(cfg, "max_steps", 0)
    passes = getattr(cfg, "passes", 0)
    if passes > 0:
        if res_buckets:
            steps_per_epoch = max(1, len(loader))
        else:
            steps_per_epoch = max(1, len(train_dataset) // cfg.global_batch_size)
        steps_from_passes = passes * steps_per_epoch
        max_steps = steps_from_passes if max_steps <= 0 else min(steps_from_passes, max_steps)
        logger.info(
            f"Passes-based: {passes} passes × {steps_per_epoch} steps/epoch = {steps_from_passes} steps (max_steps={max_steps})"
        )
    use_step_based = max_steps > 0
    if use_step_based:
        loader = repeat_epochs(loader, sampler)
        logger.info(f"Step-based training: max_steps={max_steps}, cosine LR to min_lr={getattr(cfg, 'min_lr', 1e-6)}")
    else:
        logger.info(f"Epoch-based training: epochs={cfg.epochs}")

    dashboard = None
    if rank == 0 and bool(getattr(cfg, "live_dashboard", False)):
        from utils.training.monitoring import LiveDashboard

        _dash_total = (
            max_steps
            if use_step_based
            else int(cfg.epochs) * max(1, len(train_dataset) // max(1, cfg.global_batch_size))
        )
        dashboard = LiveDashboard(_dash_total, model_name=cfg.model_name)

    # AR curriculum / order mix change the block-causal mask during training.
    ar_curriculum_mode = str(getattr(cfg, "ar_curriculum_mode", "none") or "none").strip().lower()
    ar_order_mix_spec = str(getattr(cfg, "ar_order_mix", "") or "")
    ar_order_mix_list = parse_ar_order_mix(ar_order_mix_spec)
    dynamic_ar_runtime = (ar_curriculum_mode != "none") or bool(ar_order_mix_list)

    def _ar_at(step: int) -> tuple[int, str]:
        blocks, order = resolve_ar_for_step(
            step,
            base_blocks=int(getattr(cfg, "num_ar_blocks", 0)),
            base_order=str(getattr(cfg, "ar_block_order", "raster") or "raster"),
            curriculum_mode=ar_curriculum_mode,
            warmup_steps=int(getattr(cfg, "ar_curriculum_warmup_steps", 0)),
            ramp_start=int(getattr(cfg, "ar_curriculum_ramp_start", 0)),
            ramp_end=int(getattr(cfg, "ar_curriculum_ramp_end", 0)),
            curriculum_start_blocks=int(getattr(cfg, "ar_curriculum_start_blocks", -1)),
            curriculum_target_blocks=int(getattr(cfg, "ar_curriculum_target_blocks", -1)),
            order_mix=ar_order_mix_spec,
        )
        return int(blocks), str(order)

    last_runtime_ar: tuple[int, str] | None = None
    if dynamic_ar_runtime:
        last_runtime_ar = _ar_at(start_step)
        b0, o0 = last_runtime_ar
        apply_runtime_ar(raw_model, num_ar_blocks=b0, ar_block_order=o0)
        if rank == 0:
            logger.info(
                "Dynamic AR enabled: mode=%s warmup=%d ramp=[%d,%d] mix=%s initial=(blocks=%d order=%s)",
                ar_curriculum_mode,
                int(getattr(cfg, "ar_curriculum_warmup_steps", 0)),
                int(getattr(cfg, "ar_curriculum_ramp_start", 0)),
                int(getattr(cfg, "ar_curriculum_ramp_end", 0)),
                ",".join(ar_order_mix_list) if ar_order_mix_list else "(none)",
                b0,
                o0,
            )

    train_model = model
    if dynamic_ar_runtime and cfg.use_compile:
        logger.warning("Skipping torch.compile because dynamic AR runtime is enabled (curriculum/order-mix).")
    elif cfg.use_compile:
        compile_mode = str(getattr(cfg, "compile_mode", "reduce-overhead") or "reduce-overhead")
        try:
            train_model = torch.compile(model, mode=compile_mode)
            logger.info("Model compiled with torch.compile(mode=%r)", compile_mode)
        except Exception as e:
            logger.warning(f"compile failed: {e}, continuing without compile")

    # Attention auxiliaries capture block-0 cross-attention of the VP x_t; not defined for flow/MDM inputs.
    attn_aux_supported = (
        not bool(getattr(cfg, "flow_matching_training", False))
        and not (float(getattr(cfg, "mdm_mask_ratio", 0.0)) > 0 or getattr(cfg, "mdm_mask_schedule", None))
        and hasattr(raw_model, "num_patches")
    )
    attn_aux_on = attn_aux_supported and (
        float(getattr(cfg, "attn_grounding_loss_weight", 0.0)) > 0
        or float(getattr(cfg, "attn_token_coverage_loss_weight", 0.0)) > 0
    )
    if use_ddp and (attn_aux_on or float(getattr(cfg, "bridge_aux_weight", 0.0)) > 0):
        raise ValueError(
            "--bridge-aux-weight and the attention auxiliary losses run a second forward pass per step, "
            "which DDP (one forward per backward) cannot reduce; train them on a single GPU."
        )

    steps = start_step
    log_steps = 0
    running_loss_t: torch.Tensor | None = None
    running_ag_sum_t: torch.Tensor | None = None
    running_ag_count = 0
    running_cov_sum_t: torch.Tensor | None = None
    running_cov_count = 0
    start_time = time()
    refinement_prob = getattr(cfg, "refinement_prob", 0.0)
    refinement_max_t = getattr(cfg, "refinement_max_t", 150)
    curriculum_steps = getattr(cfg, "curriculum_caption_steps", None) or []
    curriculum_lengths = getattr(cfg, "curriculum_max_lengths", None) or [300]
    save_best = getattr(cfg, "save_best", True)
    val_every = getattr(cfg, "val_every", 2000)
    early_stopping_patience = getattr(cfg, "early_stopping_patience", 0)
    use_val_for_best = val_loader is not None
    best_val_loss = float("inf")
    no_improvement_count = 0
    epoch = -1
    save_polyak = int(getattr(cfg, "save_polyak", 0) or 0)
    polyak_buf = None  # running average of the last ~save_polyak weight snapshots (rank 0)
    tb_writer = None
    if rank == 0 and getattr(cfg, "tensorboard_dir", None):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(Path(cfg.tensorboard_dir) / Path(cfg.results_dir).name))
            logger.info(f"TensorBoard logging to {cfg.tensorboard_dir}")
        except (ImportError, OSError) as e:
            logger.warning(f"TensorBoard not available: {e}")
    wandb_run = None
    if rank == 0 and getattr(cfg, "wandb_project", None):
        try:
            import wandb

            wandb_run = wandb.init(project=cfg.wandb_project, config=to_plain_dict(cfg))
            logger.info(f"WandB project: {cfg.wandb_project}")
        except Exception as e:  # missing package, login or network: train without WandB
            logger.warning(f"WandB not available: {e}")

    while True:
        if not use_step_based:
            epoch += 1
            if epoch >= cfg.epochs:
                break
            if use_ddp:
                sampler.set_epoch(epoch)

        for batch in loader:
            last_attn_grounding_t: torch.Tensor | None = None
            last_attn_cov_t: torch.Tensor | None = None
            if dynamic_ar_runtime:
                curr = _ar_at(steps)
                if curr != last_runtime_ar:
                    apply_runtime_ar(raw_model, num_ar_blocks=curr[0], ar_block_order=curr[1])
                    last_runtime_ar = curr
                    if rank == 0:
                        logger.info(
                            "AR runtime update @step=%d -> num_ar_blocks=%d ar_block_order=%s", steps, curr[0], curr[1]
                        )
            if use_step_based:
                lr = get_lr_cosine(
                    steps,
                    max_steps,
                    getattr(cfg, "lr_warmup_steps", 500),
                    cfg.lr,
                    getattr(cfg, "min_lr", 1e-6),
                )
                for g in opt.param_groups:
                    g["lr"] = lr

            _need_pixels = float(getattr(cfg, "repa_weight", 0.0)) > 0 or "latent_values" not in batch
            pixel_values = to_train_device(batch["pixel_values"], device) if _need_pixels else None
            captions = batch["captions"]
            if float(getattr(cfg, "train_originality_augment_prob", 0.0)) > 0:
                import numpy as np
                from utils.prompt.art_direction import inject_originality_tokens

                prob = float(cfg.train_originality_augment_prob)
                ost = max(0.0, min(1.0, float(getattr(cfg, "train_originality_strength", 0.5))))
                rng_np = np.random.default_rng((int(cfg.global_seed) + int(steps) * 1_000_003 + int(rank)) % (2**32))
                captions = [
                    inject_originality_tokens(cap or "", ost, rng_np) if rng_np.random() < prob else cap
                    for cap in captions
                ]
            negative_captions = batch.get("negative_captions") or [""] * len(captions)
            styles = batch.get("styles") or [""] * len(captions)

            max_caption_len = (
                get_curriculum_max_length(steps, curriculum_steps, curriculum_lengths) if curriculum_steps else 300
            )
            layout_batch = batch.get("prompt_layout")
            # Per-sample caption dropout to the empty prompt: exactly the uncond sample.py encodes for CFG,
            # through the same tokenizer/encoders/fusion (a zero vector would leave that uncond untrained).
            cap_drop_schedule = getattr(cfg, "caption_dropout_schedule", None)
            p_drop = (
                get_scheduled_caption_dropout(steps, cap_drop_schedule)
                if cap_drop_schedule
                else float(getattr(cfg, "caption_dropout_prob", 0.0) or 0.0)
            )
            if p_drop > 0:
                drop = (torch.rand(len(captions)) < p_drop).tolist()
                captions = ["" if d else c for c, d in zip(captions, drop, strict=True)]
                if layout_batch is not None:
                    layout_batch = [None if d else spec for spec, d in zip(layout_batch, drop, strict=True)]

            with torch.amp.autocast("cuda", enabled=cfg.use_bf16, dtype=torch.bfloat16):
                with torch.no_grad():
                    if "latent_values" in batch:
                        latents = to_train_device(batch["latent_values"], device, dtype=torch.bfloat16)
                    else:
                        latents = encode_images_vae(pixel_values, vae, effective_latent_scale)
                    # Img2img training: sometimes start from the row's init image so the model learns to edit it.
                    init_pixel_values = batch.get("init_pixel_values")
                    if init_pixel_values is not None and getattr(cfg, "img2img_prob", 0) > 0:
                        use_init = torch.rand(latents.shape[0], device=device, dtype=torch.bfloat16) < cfg.img2img_prob
                        if use_init.any():
                            nbc = device.type == "cuda"
                            idx = use_init.nonzero(as_tuple=False).view(-1)
                            init_lat_subset = encode_images_vae(
                                init_pixel_values.to(device, non_blocking=nbc)[idx],
                                vae,
                                effective_latent_scale,
                            )
                            latents = latents.clone()
                            latents[idx] = init_lat_subset
                    edit_latents = edit_source_latents(
                        cfg,
                        batch,
                        vae,
                        effective_latent_scale,
                        device,
                        dropout=float(getattr(cfg, "edit_source_dropout", 0.1)),
                    )
                    train_pe = bool(getattr(cfg, "train_prompt_emphasis", False))
                    pos_caps = captions
                    token_weights = None
                    if train_pe:
                        from utils.prompt.text import batch_encoder_token_weights

                        pos_caps, token_weights = batch_encoder_token_weights(
                            captions,
                            tokenizer,
                            max_caption_len,
                            device=device,
                            dtype=torch.bfloat16,
                            text_bundle=text_bundle,
                        )
                    # Frozen encoders run without grad inside the bundle; the fusion layers need it to train.
                    with torch.set_grad_enabled(fusion is not None):
                        encoder_hidden = encode_text(
                            pos_caps,
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                            **encode_kw_for_batch(pos_caps, text_bundle, layout_specs=layout_batch),
                        )
                    neg_caps = negatives_strip_emphasis(negative_captions, train_pe)

                    def _enc_batch(caps: list, max_caption_len: int = max_caption_len) -> torch.Tensor:
                        return encode_text(
                            caps,
                            tokenizer,
                            text_encoder,
                            device,
                            dtype=torch.bfloat16,
                            max_length=max_caption_len,
                            text_bundle=text_bundle,
                            train_fusion=True,
                            **encode_kwargs_for_captions(caps, text_bundle),
                        )

                    encoder_hidden_neg, style_embedding = encode_neg_and_style_embeddings(
                        neg_caps,
                        styles,
                        style_embed_dim=int(getattr(cfg, "style_embed_dim", 0) or 0),
                        encode_fn=_enc_batch,
                        batch_text_encode=bool(getattr(cfg, "batch_text_encode", True)),
                    )
                latents_raw = latents
                if rae_bridge is not None and latents_raw.shape[1] != 4:
                    latents = rae_bridge.rae_to_dit(latents_raw)
                    z_cycle = latents_raw.detach()
                else:
                    z_cycle = None
                t = step_timesteps(cfg, diffusion.num_timesteps, latents.shape[0], device, global_step=steps)
                model_kwargs = conditioning_kwargs(
                    cfg,
                    batch,
                    latents,
                    device,
                    encoder_hidden=encoder_hidden,
                    encoder_hidden_neg=encoder_hidden_neg,
                    style_embedding=style_embedding,
                    token_weights=token_weights,
                    creativity_jitter_std=float(getattr(cfg, "creativity_jitter_std", 0.0)),
                    edit_latents=edit_latents,
                )
                sample_weights = batch.get("sample_weights")
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device, non_blocking=device.type == "cuda")
                curriculum_diff_steps = getattr(cfg, "curriculum_difficulty_steps", None) or []
                if curriculum_diff_steps and "difficulty" in batch:
                    diff = batch["difficulty"].to(device)
                    easy_first = getattr(cfg, "curriculum_difficulty_easy_first", True)
                    if steps < curriculum_diff_steps[0]:
                        w = 1.0 - diff if easy_first else 1.0 + diff
                    elif len(curriculum_diff_steps) >= 2 and steps >= curriculum_diff_steps[-1]:
                        w = 1.0 + diff if easy_first else 1.0 - diff
                    else:
                        w = torch.ones_like(diff)
                    if sample_weights is not None:
                        sample_weights = sample_weights * w
                    else:
                        sample_weights = w
                noise = training_noise(cfg, latents)
                attn_gw = float(getattr(cfg, "attn_grounding_loss_weight", 0.0))
                gv = batch.get("grounding_mask_valid")
                has_mask_supervision = "grounding_mask" in batch and (gv is None or bool(gv.any()))
                use_attn_grounding = attn_gw > 0.0 and has_mask_supervision and attn_aux_supported
                cov_w = float(getattr(cfg, "attn_token_coverage_loss_weight", 0.0))
                use_attn_coverage = cov_w > 0.0 and attn_aux_supported
                use_attn_aux = use_attn_grounding or use_attn_coverage

                loss = diffusion_step_loss(
                    cfg,
                    diffusion,
                    train_model,
                    latents,
                    t,
                    model_kwargs,
                    sample_weights,
                    noise,
                    refinement_prob=0.0 if use_attn_aux else refinement_prob,
                    refinement_max_t=refinement_max_t,
                )
                # Per-forward attributes of the main pass; read before any other forward overwrites them.
                moe_aux_loss = getattr(raw_model, "moe_aux_loss", None)
                repa_pred = getattr(raw_model, "_repa_projected", None)
                if use_attn_aux:
                    from utils.training.objectives import (
                        capture_dit_block0_cross_attn,
                        grounding_loss_from_attn,
                        token_coverage_loss_from_attn,
                    )

                    attn_w = capture_dit_block0_cross_attn(
                        train_model=train_model,
                        diffusion=diffusion,
                        latents_bchw=latents,
                        t=t,
                        model_kwargs=model_kwargs,
                        training_noise=noise,
                    )
                    if use_attn_grounding:
                        nbc = device.type == "cuda"
                        gm = batch["grounding_mask"].to(device=device, dtype=torch.float32, non_blocking=nbc)
                        gv_t = batch.get("grounding_mask_valid")
                        ag = grounding_loss_from_attn(
                            attn_w=attn_w,
                            train_model=raw_model,
                            grounding_mask_b1hw=gm,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            sample_valid=gv_t,
                            min_fg_patch_mass=float(getattr(cfg, "attn_grounding_min_fg_patch_mass", 0.0)),
                        )
                        loss = loss + attn_gw * ag
                        last_attn_grounding_t = ag.detach()
                    if use_attn_coverage:
                        cov = token_coverage_loss_from_attn(
                            attn_w=attn_w,
                            token_start=int(getattr(cfg, "attn_grounding_token_start", 0)),
                            token_end=int(getattr(cfg, "attn_grounding_token_end", 0)),
                            target_coverage=float(getattr(cfg, "attn_token_coverage_target", 0.025)),
                            sample_valid=None,
                            token_weights=token_weights,
                        )
                        loss = loss + cov_w * cov
                        last_attn_cov_t = cov.detach()
                moe_w = float(getattr(cfg, "moe_balance_loss_weight", 0.0))
                if moe_w > 0 and moe_aux_loss is not None:
                    loss = loss + moe_w * moe_aux_loss.to(loss.dtype)
                repa_w = float(getattr(cfg, "repa_weight", 0.0))
                if repa_w > 0 and pixel_values is not None and repa_pred is not None:
                    loss = loss + repa_w * repa_alignment_loss(repa_pred, repa_features(pixel_values, device, cfg))
                loss = loss + bridge_aux_loss(cfg, diffusion, train_model, latents, model_kwargs, global_step=steps)
                if rae_bridge is not None and z_cycle is not None:
                    cw = float(getattr(cfg, "rae_bridge_cycle_weight", 0.0))
                    if cw > 0:
                        loss = loss + cw * rae_bridge.cycle_loss(z_cycle)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (steps + 1) % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(opt_params, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                decay = ema_decay_at((steps + 1) // cfg.grad_accum_steps, cfg.ema_decay)
                for live, shadow in ema_pairs:
                    update_ema(shadow, live, decay)
                if save_polyak > 0 and rank == 0 and steps % max(1, int(cfg.log_every)) == 0:
                    sd = raw_model.state_dict()
                    if polyak_buf is None:
                        polyak_buf = {k: v.detach().cpu().clone() for k, v in sd.items()}
                    else:
                        ema_update_state(polyak_buf, sd, 1.0 - 1.0 / save_polyak)

            if running_loss_t is None:
                running_loss_t = torch.zeros((), device=device)
            running_loss_t = running_loss_t + loss.detach() * cfg.grad_accum_steps
            if last_attn_grounding_t is not None:
                if running_ag_sum_t is None:
                    running_ag_sum_t = torch.zeros((), device=device)
                running_ag_sum_t = running_ag_sum_t + last_attn_grounding_t.float()
                running_ag_count += 1
            if last_attn_cov_t is not None:
                if running_cov_sum_t is None:
                    running_cov_sum_t = torch.zeros((), device=device)
                running_cov_sum_t = running_cov_sum_t + last_attn_cov_t.float()
                running_cov_count += 1
            log_steps += 1
            steps += 1

            if steps % cfg.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed if elapsed > 0 else 0
                avg_loss = (running_loss_t / log_steps).item() if running_loss_t is not None else 0.0
                if use_ddp:
                    loss_sum = torch.tensor([avg_loss], device=device)
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    avg_loss = loss_sum.item() / world_size
                lr_val = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                lr_str = f" lr={lr_val:.2e}" if use_step_based else ""
                scalars = {"loss": avg_loss, "lr": lr_val}
                aux_bits = []
                if running_ag_count > 0 and running_ag_sum_t is not None:
                    scalars["attn_grounding_aux"] = (running_ag_sum_t / running_ag_count).item()
                    aux_bits.append(f"attn_grnd={scalars['attn_grounding_aux']:.4f}")
                if running_cov_count > 0 and running_cov_sum_t is not None:
                    scalars["attn_token_coverage_aux"] = (running_cov_sum_t / running_cov_count).item()
                    aux_bits.append(f"attn_cov={scalars['attn_token_coverage_aux']:.4f}")
                aux_str = "".join(f" {b}" for b in aux_bits)
                logger.info(
                    f"step={steps:07d} epoch={epoch} loss={avg_loss:.4f} steps/s={steps_per_sec:.2f}{lr_str}{aux_str}"
                )
                if dashboard is not None:
                    dashboard.update(
                        steps,
                        avg_loss,
                        steps_per_sec,
                        lr_val,
                        img_per_sec=steps_per_sec * cfg.global_batch_size,
                        aux="  ".join(aux_bits),
                    )
                if tb_writer is not None:
                    for name, value in scalars.items():
                        tb_writer.add_scalar(name, value, steps)
                if wandb_run is not None:
                    wandb_run.log({**scalars, "step": steps}, step=steps)
                log_images_every = getattr(cfg, "log_images_every", 0)
                if (
                    rank == 0
                    and log_images_every > 0
                    and steps > 0
                    and steps % log_images_every == 0
                    and (tb_writer is not None or wandb_run is not None)
                ):
                    try:
                        log_sample_image(
                            ema,
                            diffusion,
                            tokenizer,
                            text_encoder,
                            vae,
                            device,
                            cfg,
                            steps,
                            tb_writer,
                            rae_bridge=rae_bridge,
                            text_bundle=text_bundle,
                            wandb_run=wandb_run,
                        )
                    except Exception as e:  # a failed preview must not stop training
                        logger.warning(f"Log sample image failed: {e}")
                # Save best by train loss only when not using validation (when using val, best is by val loss)
                if save_best and rank == 0 and not use_val_for_best and avg_loss < best_loss:
                    best_loss = avg_loss
                    path = ckpt_dir / "best.pt"
                    torch.save(_checkpoint_dict(steps, best_loss=best_loss), path)
                    _maybe_save_lora_adapter(raw_model, path)
                    logger.info(f"Best checkpoint saved: {path} (train loss={best_loss:.4f})")
                if running_loss_t is not None:
                    running_loss_t.zero_()
                running_ag_sum_t = None
                running_ag_count = 0
                running_cov_sum_t = None
                running_cov_count = 0
                log_steps = 0
                start_time = time()

            if steps > 0 and steps % cfg.ckpt_every == 0 and rank == 0:
                path = ckpt_dir / f"{steps:07d}.pt"
                torch.save(_checkpoint_dict(steps), path)
                _maybe_save_lora_adapter(raw_model, path)
                logger.info(f"Checkpoint saved: {path}")
                if polyak_buf is not None:
                    polyak_ckpt = {"model": polyak_buf, "ema": polyak_buf, "steps": steps, "config": cfg}
                    torch.save(polyak_ckpt, ckpt_dir / "polyak.pt")
                    logger.info(f"Polyak checkpoint saved: {ckpt_dir / 'polyak.pt'}")

            # Validation + early stopping: save best by val loss, stop when no improvement
            should_stop = False
            if val_loader is not None and steps > 0 and steps % val_every == 0 and rank == 0:
                val_loss = eval_val_loss(
                    ema,
                    diffusion,
                    val_loader,
                    tokenizer,
                    text_encoder,
                    vae,
                    device,
                    cfg,
                    refinement_prob=refinement_prob,
                    refinement_max_t=refinement_max_t,
                    max_caption_len=curriculum_lengths[-1] if curriculum_lengths else 300,
                    max_batches=getattr(cfg, "val_max_batches", None),
                    rae_bridge=rae_bridge,
                    latent_scale=effective_latent_scale,
                    text_bundle=text_bundle,
                    global_step=steps,
                )
                logger.info(f"Val loss @ step {steps}: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    path = ckpt_dir / "best.pt"
                    torch.save(_checkpoint_dict(steps, best_loss=best_val_loss), path)
                    _maybe_save_lora_adapter(raw_model, path)
                    logger.info(f"Best checkpoint saved: {path} (val loss={best_val_loss:.4f})")
                else:
                    no_improvement_count += 1
                    logger.info(f"No val improvement ({no_improvement_count}/{early_stopping_patience})")
                if early_stopping_patience > 0 and no_improvement_count >= early_stopping_patience:
                    should_stop = True
                    logger.info(f"Early stopping: no val improvement for {early_stopping_patience} checks")
            if use_ddp and steps > 0 and steps % val_every == 0:
                stop_tensor = torch.tensor([1 if (rank == 0 and should_stop) else 0], device=device, dtype=torch.long)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1
            if should_stop:
                break

            if use_step_based and steps >= max_steps:
                break
        if use_step_based and steps >= max_steps:
            break

    if dashboard is not None:
        dashboard.close()
    if use_ddp:
        dist.destroy_process_group()
    logger.info("Training done.")


if __name__ == "__main__":
    from utils.runtime import consume_profile_args, run_with_cprofile

    _argv, _pcfg = consume_profile_args(sys.argv)
    sys.argv = _argv

    def _train_entry() -> None:
        parser = build_train_arg_parser()
        args = parser.parse_args()
        apply_book_train_preset_to_args(args)
        cfg = build_train_config_from_args(args)
        main(cfg)

    if _pcfg is not None:
        run_with_cprofile(_train_entry, _pcfg)
    else:
        _train_entry()
