#!/usr/bin/env python3
"""
Train ``ReferenceTokenProjector`` (IP-Adapter-style tokens) on (image, caption) pairs.

Freezes any DiT; only the small projector is optimized. Saves a ``.pt`` usable with
``sample.py --reference-adapter-pt``.

    python -m scripts.tools train_reference_adapter \\
        --data-dir data/smoke --out checkpoints/reference_adapter.pt --steps 200

Expects image files with matching ``.txt`` captions (or a JSONL with
``image_path`` / ``caption`` keys).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, str]], size: int = 224):
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, caption = self.pairs[idx]
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        arr = torch.from_numpy(__import__("numpy").asarray(img).astype("float32") / 255.0).permute(2, 0, 1)
        return arr, caption


def _collect_pairs(data_dir: Path | None, jsonl: Path | None) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    if jsonl and jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            p = Path(str(row.get("image_path") or row.get("image") or ""))
            cap = str(row.get("caption") or row.get("prompt") or "").strip()
            if p.is_file() and cap:
                pairs.append((p, cap))
    if data_dir and data_dir.is_dir():
        for img in sorted(data_dir.rglob("*")):
            if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            txt = img.with_suffix(".txt")
            if not txt.is_file():
                continue
            cap = txt.read_text(encoding="utf-8").strip()
            if cap:
                pairs.append((img, cap))
    return pairs


def _cheap_image_embed(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic pooled 'CLIP-like' stub when transformers CLIP is unavailable."""
    # Global average pool + fixed projection via hash of spatial mean
    pooled = x.mean(dim=(2, 3))  # B,3
    # Expand with sinusoidal features of channel means
    B = pooled.shape[0]
    t = torch.linspace(0, 1, dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)
    rgb = pooled.mean(dim=1, keepdim=True)
    return torch.tanh(t * 3.0 + rgb * 2.0)


@torch.no_grad()
def _try_clip_embed(images: torch.Tensor, captions: list[str], device: torch.device, dim: int):
    try:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # images are 0-1 CHW
        pil = []
        import numpy as np

        for i in range(images.shape[0]):
            arr = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil.append(Image.fromarray(arr))
        inputs = proc(text=captions, images=pil, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        img_e = out.image_embeds
        txt_e = out.text_embeds
        return img_e, txt_e, img_e.shape[-1]
    except Exception:
        img_e = _cheap_image_embed(images.to(device), dim)
        # Caption hash embed
        rows = []
        for c in captions:
            h = abs(hash(c)) % (10**8)
            g = torch.Generator(device=device)
            g.manual_seed(h % (2**31 - 1))
            rows.append(torch.randn(dim, generator=g, device=device))
        txt_e = torch.stack(rows, dim=0)
        txt_e = F.normalize(txt_e, dim=-1)
        img_e = F.normalize(img_e, dim=-1)
        return img_e, txt_e, dim


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ReferenceTokenProjector")
    ap.add_argument("--data-dir", type=str, default="", help="Folder of image+txt pairs")
    ap.add_argument("--jsonl", type=str, default="", help="JSONL with image_path/caption")
    ap.add_argument("--out", type=str, required=True, help="Output .pt path")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip-dim", type=int, default=512)
    ap.add_argument("--hidden-size", type=int, default=768)
    ap.add_argument("--num-tokens", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    pairs = _collect_pairs(
        Path(args.data_dir) if args.data_dir else None,
        Path(args.jsonl) if args.jsonl else None,
    )
    if len(pairs) < 2:
        print("Need at least 2 (image, caption) pairs.", file=sys.stderr)
        return 2

    from models.reference_token_projection import ReferenceTokenProjector

    device = torch.device(args.device)
    ds = _PairDataset(pairs)
    # Drop last incomplete only if batch would be empty
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, drop_last=False)

    # Probe embed dim
    sample_imgs, sample_caps = next(iter(loader))
    img_e, txt_e, clip_dim = _try_clip_embed(sample_imgs, list(sample_caps), device, args.clip_dim)

    proj = ReferenceTokenProjector(clip_dim, args.hidden_size, num_tokens=args.num_tokens).to(device)
    # Text-side target: project caption embed to same token space (shared projector for contrastive)
    text_proj = ReferenceTokenProjector(clip_dim, args.hidden_size, num_tokens=args.num_tokens).to(device)
    opt = torch.optim.AdamW(list(proj.parameters()) + list(text_proj.parameters()), lr=args.lr)

    step = 0
    it = iter(loader)
    while step < args.steps:
        try:
            imgs, caps = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, caps = next(it)
        imgs = imgs.to(device)
        img_e, txt_e, _ = _try_clip_embed(imgs, list(caps), device, clip_dim)
        img_tok = proj(img_e)
        txt_tok = text_proj(txt_e)
        # InfoNCE on mean-pooled tokens
        a = F.normalize(img_tok.mean(dim=1), dim=-1)
        b = F.normalize(txt_tok.mean(dim=1), dim=-1)
        logits = a @ b.T / 0.07
        labels = torch.arange(a.shape[0], device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 20 == 0 or step == 1:
            print(f"step={step}/{args.steps} loss={float(loss):.4f}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": proj.state_dict(),
            "clip_dim": int(clip_dim),
            "hidden_size": int(args.hidden_size),
            "num_tokens": int(args.num_tokens),
        },
        out,
    )
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
