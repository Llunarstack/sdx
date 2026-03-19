"""
Precompute VAE latents for a dataset and save to disk. Then train with --latent-cache-dir for faster training.
Usage: python scripts/precompute_latents.py --data-path /path/to/images --out-dir /path/to/latent_cache --image-size 256
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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
    def __init__(self, data_path, image_size=256):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.paths = []
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
                        self.paths.append(p)
        else:
            for subdir in self.data_path.iterdir():
                if not subdir.is_dir():
                    continue
                for p in subdir.glob("*"):
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        self.paths.append(str(p))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        pil = Image.open(p).convert("RGB")
        pil = center_crop(pil, self.image_size)
        img = np.array(pil).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True, help="Latent cache directory (e.g. latent_cache)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE or RAE model id/path (uses --autoencoder-type)")
    parser.add_argument("--scale", type=float, default=0.18215)
    parser.add_argument(
        "--autoencoder-type",
        type=str,
        default="kl",
        choices=["kl", "rae"],
        help="kl=AutoencoderKL, rae=AutoencoderRAE",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffusers import AutoencoderKL, AutoencoderRAE
    if args.autoencoder_type == "rae":
        vae = AutoencoderRAE.from_pretrained(args.vae).to(device).eval()
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
        vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()
        latent_scale = args.scale

    dataset = ImagePaths(args.data_path, args.image_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for i, (imgs, paths) in enumerate(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            enc = vae.encode(imgs)
            if hasattr(enc, "latent_dist"):
                latents = enc.latent_dist.sample() * latent_scale
            else:
                latents = enc.latent
        for j, p in enumerate(paths):
            name = Path(p).stem + ".pt"
            torch.save(latents[j].cpu(), out_dir / name)
        if (i + 1) % 100 == 0:
            print(f"Processed {(i+1)*args.batch_size} / {len(dataset)}")
    print(f"Done. Latents saved to {out_dir}. Train with: --latent-cache-dir {out_dir}")


if __name__ == "__main__":
    main()
