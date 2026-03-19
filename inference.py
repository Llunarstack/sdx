"""
Inference with optional refinement: fix imperfections during/after generation.
Set allow_imperfect_output=True (or refine_output=False) when you want the raw/fucked look.
"""
import argparse
from pathlib import Path
import sys
import torch

# Run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_dit_build_kwargs
from diffusion import create_diffusion
from models import DiT_models_text


def load_model_from_ckpt(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint must contain config")
    model_name = getattr(cfg, "model_name", "DiT-XL/2-Text")
    model_fn = DiT_models_text.get(model_name) or DiT_models_text["DiT-XL/2-Text"]
    model = model_fn(**get_dit_build_kwargs(cfg, class_dropout_prob=0.0))
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, cfg


def refine_latent_once(diffusion, model, x_0_latent, encoder_hidden_states, t_refine=50, device="cuda"):
    """
    Add a small amount of noise to x_0 then denoise once to fix artifacts.
    User can skip this by setting allow_imperfect_output=True.
    """
    with torch.no_grad():
        t = torch.full((x_0_latent.shape[0],), t_refine, device=device, dtype=torch.long)
        noise = torch.randn_like(x_0_latent, device=device)
        diffusion._to_device(device)
        sqrt_alpha = diffusion.sqrt_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_0_latent.ndim - 1)]
        sqrt_one = diffusion.sqrt_one_minus_alpha_cumprod.to(device)[t][(...,) + (None,) * (x_0_latent.ndim - 1)]
        x_t = sqrt_alpha * x_0_latent + sqrt_one * noise
        x_0_pred, _ = diffusion.p_step(
            model, x_t, t, model_kwargs={"encoder_hidden_states": encoder_hidden_states}
        )
    return x_0_pred


def main():
    parser = argparse.ArgumentParser(description="Load checkpoint and run one sample (refinement optional)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. results/.../best.pt)")
    parser.add_argument("--refine-output", action="store_true", default=True, help="Run one refinement pass to fix imperfections (default: True)")
    parser.add_argument("--allow-imperfect", action="store_true", help="Skip refinement; output raw result (user wants fucked look)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    refine = args.refine_output and not args.allow_imperfect
    print(f"Refinement pass: {'enabled (fix imperfections)' if refine else 'disabled (allow imperfect output)'}")

    model, cfg = load_model_from_ckpt(args.ckpt, args.device)
    diffusion = create_diffusion(
        timestep_respacing=getattr(cfg, "timestep_respacing", ""),
        num_timesteps=getattr(cfg, "num_timesteps", 1000),
        beta_schedule=getattr(cfg, "beta_schedule", "linear"),
        prediction_type=getattr(cfg, "prediction_type", "epsilon"),
    )
    print("Model and diffusion loaded. Use this module from your sampling script:")
    print("  refine_latent_once(diffusion, model, x_0_latent, encoder_hidden_states, t_refine=50)")
    print("  Set allow_imperfect_output=True in config or --allow-imperfect to skip refinement.")


if __name__ == "__main__":
    main()
