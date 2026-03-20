#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
    from ViT.dataset import text_feature_vector
    from ViT.model import build_vit_model

    p = argparse.ArgumentParser(description="Export ViT fused embeddings to .npz for retrieval/reranking")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_vit_model(
        model_name=cfg.get("model_name", "vit_base_patch16_224"),
        pretrained=False,
        text_feat_dim=int(cfg.get("text_feat_dim", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
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
    vecs = []
    paths = []
    with inp.open("r", encoding="utf-8") as rf:
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
            with torch.no_grad():
                out = model(x, txt)
                emb = out["embedding"].squeeze(0).detach().cpu().numpy()
            vecs.append(emb.astype(np.float32))
            paths.append(str(p_img))

    if not vecs:
        print("No valid rows for embedding export.", file=sys.stderr)
        return 2

    arr = np.stack(vecs, axis=0)
    outp = Path(args.out_npz)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, embeddings=arr, image_paths=np.array(paths, dtype=object))
    print(f"[ViT] exported {arr.shape[0]} embeddings ({arr.shape[1]} dim) -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

