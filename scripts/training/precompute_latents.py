"""
Precompute VAE latents for a dataset and save to disk. Then train with --latent-cache-dir for faster training.
Usage: python scripts/training/precompute_latents.py --data-path /path/to/images --out-dir /path/to/latent_cache --image-size 256
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def center_crop(pil_image, image_size: int):
    w, h = pil_image.size
    while min(w, h) >= 2 * image_size:
        pil_image = pil_image.resize((w // 2, h // 2), resample=Image.BOX)
        w, h = pil_image.size
    scale = image_size / min(w, h)
    pil_image = pil_image.resize((round(w * scale), round(h * scale)), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    h, w = arr.shape[:2]
    crop_y = (h - image_size) // 2
    crop_x = (w - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class ImagePaths(Dataset):
    def __init__(self, data_path, image_size=256, *, data_root: str | None = None, out_dir: Path | None = None):
        self.data_path = Path(data_path)
        self.data_root = Path(data_root) if data_root else None
        self.image_size = image_size
        self.out_dir = out_dir
        self.paths: list[str] = []
        if self.data_path.suffix.lower() == ".jsonl":
            import json

            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    p = d.get("image_path") or d.get("path") or d.get("image")
                    if p:
                        self.paths.append(str(p))
        else:
            for subdir in self.data_path.iterdir():
                if not subdir.is_dir():
                    continue
                images_dir = subdir / "images"
                scan = images_dir if images_dir.is_dir() else subdir
                for p in scan.glob("*"):
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        self.paths.append(str(p))

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_file():
            return p.resolve()
        if self.data_root is not None:
            cand = (self.data_root / p).resolve()
            if cand.is_file():
                return cand
        if self.data_path.suffix.lower() == ".jsonl":
            cand = (self.data_path.parent / p).resolve()
            if cand.is_file():
                return cand
        return p.resolve()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        resolved = self._resolve(rel)
        if self.out_dir is not None:
            cache = self.out_dir / (resolved.stem + ".pt")
            if cache.is_file():
                return None, str(resolved)
        pil = Image.open(resolved).convert("RGB")
        pil = center_crop(pil, self.image_size)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, str(resolved)


def _collate_skip(batch):
    imgs, paths = [], []
    for item in batch:
        if item[0] is None:
            continue
        imgs.append(item[0])
        paths.append(item[1])
    if not imgs:
        return None, paths
    return torch.stack(imgs), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Image folder or legacy alias.")
    parser.add_argument("--manifest-jsonl", type=str, default=None, help="Training manifest (preferred).")
    parser.add_argument("--data-root", type=str, default=None, help="Resolve manifest image_path (e.g. /workspace/data).")
    parser.add_argument("--out-dir", type=str, required=True, help="Latent cache directory (e.g. latent_cache)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--vae",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE or RAE model id/path (uses --autoencoder-type)",
    )
    parser.add_argument("--scale", type=float, default=0.18215)
    parser.add_argument(
        "--autoencoder-type",
        type=str,
        default="kl",
        choices=["kl", "rae"],
        help="kl=AutoencoderKL, rae=AutoencoderRAE",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers; -1 = min(8, CPU count). Use 0 on Windows if workers hang.",
    )
    args = parser.parse_args()
    manifest = args.manifest_jsonl or args.data_path
    if not manifest:
        parser.error("Provide --manifest-jsonl or --data-path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = args.num_workers
    if nw < 0:
        nw = min(16, (os.cpu_count() or 1))
    from utils.modeling.autoencoder_loading import get_autoencoder_class

    if args.autoencoder_type == "rae":
        vae = get_autoencoder_class("rae").from_pretrained(args.vae).to(device).eval()
        latent_scale = 1.0
        ae_cfg = getattr(vae, "config", None)
        latent_channels_rae = getattr(ae_cfg, "encoder_hidden_size", None) if ae_cfg is not None else None
        if latent_channels_rae is not None and int(latent_channels_rae) != 4:
            print(
                f"Warning: RAE latents have {latent_channels_rae} channels, but this repo's DiT expects 4-channel SD latents. "
                "Precomputing may not be usable until the DiT/diffusion latent dimensions are updated.",
                file=sys.stderr,
            )
    else:
        vae = get_autoencoder_class("kl").from_pretrained(args.vae).to(device).eval()
        latent_scale = args.scale

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = ImagePaths(manifest, args.image_size, data_root=args.data_root, out_dir=out_dir)

    from torch.utils.data import DataLoader

    pin = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
        collate_fn=_collate_skip,
    )
    processed = skipped = 0
    for i, batch in enumerate(loader):
        imgs, paths = batch
        if imgs is None:
            skipped += len(paths)
            continue
        imgs = imgs.to(device, non_blocking=pin)
        with torch.no_grad():
            enc = vae.encode(imgs)
            if hasattr(enc, "latent_dist"):
                latents = enc.latent_dist.sample() * latent_scale
            else:
                latents = enc.latent
        for j, p in enumerate(paths):
            name = Path(p).stem + ".pt"
            torch.save(latents[j].cpu(), out_dir / name)
        processed += len(paths)
        if (i + 1) % 100 == 0:
            print(f"Encoded {processed} new, skipped {skipped} cached / {len(dataset)} total")
    print(f"Done. Encoded {processed} new ({skipped} already cached). Train with: --latent-cache-dir {out_dir}")


if __name__ == "__main__":
    main()
