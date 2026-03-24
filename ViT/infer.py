#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(image_path: str, manifest_path: Path) -> Path:
    p = Path(image_path)
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    return p


def main() -> int:
    from utils.architecture.ar_dit_vit import ar_conditioning_vector, parse_num_ar_blocks_from_row

    from ViT.checkpoint_utils import load_vit_quality_checkpoint
    from ViT.dataset import text_feature_vector
    from ViT.tta import tta_predict

    p = argparse.ArgumentParser(description="Run ViT quality/adherence scoring on JSONL manifest")
    p.add_argument("--ckpt", required=True, help="Path to ViT checkpoint (best.pt/last.pt)")
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--out", required=True, help="Output JSONL with appended quality/adherence scores")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tta", action="store_true", help="Use test-time augmentation (flip) and average predictions")
    p.add_argument("--use-ema", action="store_true", help="If checkpoint contains ema_state_dict, use it")
    p.add_argument(
        "--default-num-ar-blocks",
        type=int,
        default=None,
        help="When AR conditioning is on and a row has no AR field, use this (0/2/4) instead of 'unknown'",
    )
    args = p.parse_args()

    model, cfg = load_vit_quality_checkpoint(args.ckpt, use_ema=args.use_ema)
    use_ar = bool(cfg.get("use_ar_conditioning", False))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    image_size = int(cfg.get("image_size", 224))
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    inp = Path(args.manifest_jsonl)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open("r", encoding="utf-8") as rf, outp.open("w", encoding="utf-8") as wf:
        for line in rf:
            t = line.strip()
            if not t:
                continue
            try:
                row = json.loads(t)
            except Exception:
                continue
            image_path = row.get("image_path") or row.get("path") or row.get("image")
            caption = row.get("caption") or row.get("text") or ""
            if not isinstance(image_path, str) or not image_path:
                continue
            p_img = _resolve_path(image_path, inp)
            if not p_img.exists():
                continue
            try:
                img = Image.open(p_img).convert("RGB")
            except Exception:
                continue

            x = tfm(img).unsqueeze(0).to(device)
            txt = text_feature_vector(str(caption)).unsqueeze(0).to(device)
            ar_b = parse_num_ar_blocks_from_row(row)
            if use_ar and ar_b == -1 and args.default_num_ar_blocks is not None:
                from utils.architecture.ar_dit_vit import normalize_num_ar_blocks

                ar_b = normalize_num_ar_blocks(args.default_num_ar_blocks)
            ar_vec = (
                ar_conditioning_vector(ar_b, device=device, dtype=txt.dtype).unsqueeze(0)
                if use_ar
                else None
            )
            with torch.no_grad():
                if args.tta:
                    out = tta_predict(model, x, txt, ar_vec)
                else:
                    out = model(x, txt, ar_vec)
                quality_prob = torch.sigmoid(out["quality_logit"]).item()
                adherence = out["adherence_score"].item()

            row["vit_quality_prob"] = float(quality_prob)
            row["vit_adherence_score"] = float(adherence)
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"[ViT] wrote {n} rows -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
