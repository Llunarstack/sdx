from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- CONFIGURATION ---
SAVE_PATH = Path(r"C:\Users\macfa\Desktop\Development\sdx\docs\assets")
FILE_NAME = "design.png"
ALT_FILE_NAME = "sdx_model_architecture.png"
SCALE = 2


Color = Tuple[int, int, int]


class Theme:
    BG = (245, 248, 252)
    CARD_BG = (255, 255, 255)
    SHADOW = (15, 23, 42, 30)
    TEXT_MAIN = (15, 23, 42)
    TEXT_SUB = (71, 85, 105)
    TEXT_FADE = (100, 116, 139)
    FLOW = (30, 41, 59)
    FLOW_OPTIONAL = (124, 140, 166)

    PIXEL = ((255, 241, 242), (225, 29, 72))
    LATENT = ((240, 253, 244), (22, 163, 74))
    COND = ((238, 242, 255), (79, 70, 229))
    TRAIN = ((254, 249, 195), (202, 138, 4))
    INFER = ((224, 242, 254), (2, 132, 199))
    CORE = ((248, 250, 252), (148, 163, 184))
    NATIVE = ((236, 253, 245), (5, 150, 105))


class ProRenderer:
    def __init__(self, draw: ImageDraw.ImageDraw, shadow_layer: Image.Image):
        self.draw = draw
        self.shadow_layer = shadow_layer

    def font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        px = int(size * SCALE)
        names = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
        for name in names:
            try:
                return ImageFont.truetype(name, px)
            except Exception:
                continue
        return ImageFont.load_default()

    def card(
        self,
        box: Tuple[int, int, int, int],
        fill: Color,
        outline: Color,
        *,
        radius: int = 12,
        shadow: bool = True,
        width: int = 2,
    ) -> None:
        x0, y0, x1, y1 = [v * SCALE for v in box]
        r = radius * SCALE
        w = width * SCALE
        if shadow:
            off = 4 * SCALE
            ImageDraw.Draw(self.shadow_layer).rounded_rectangle(
                [x0 + off, y0 + off, x1 + off, y1 + off], radius=r, fill=Theme.SHADOW
            )
        self.draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=outline, width=w)

    def text(
        self,
        xy: Tuple[float, float],
        text: str,
        *,
        size: int = 18,
        color: Color = Theme.TEXT_MAIN,
        bold: bool = False,
        align: str = "center",
    ) -> None:
        font = self.font(size, bold)
        x, y = xy[0] * SCALE, xy[1] * SCALE
        anchor = "mm" if align == "center" else ("rm" if align == "right" else "lm")
        self.draw.multiline_text((x, y), text, fill=color, font=font, anchor=anchor, align=align, spacing=6 * SCALE)

    def arrow(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        *,
        color: Color = Theme.FLOW,
        width: int = 2,
        dashed: bool = False,
    ) -> None:
        sx, sy = start[0] * SCALE, start[1] * SCALE
        ex, ey = end[0] * SCALE, end[1] * SCALE

        if dashed:
            self._dashed_line((sx, sy), (ex, ey), color, width * SCALE)
        else:
            self.draw.line((sx, sy, ex, ey), fill=color, width=width * SCALE)

        # Arrow head
        import math

        angle = math.atan2(ey - sy, ex - sx)
        ah = 10 * SCALE
        p1 = (ex, ey)
        p2 = (ex - ah * math.cos(angle - 0.45), ey - ah * math.sin(angle - 0.45))
        p3 = (ex - ah * math.cos(angle + 0.45), ey - ah * math.sin(angle + 0.45))
        self.draw.polygon([p1, p2, p3], fill=color)

    def _dashed_line(self, start, end, color, width):
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        dist = max((dx * dx + dy * dy) ** 0.5, 1.0)
        ux, uy = dx / dist, dy / dist
        dash = 12 * SCALE
        gap = 8 * SCALE
        pos = 0.0
        while pos < dist:
            npos = min(pos + dash, dist)
            x0 = sx + ux * pos
            y0 = sy + uy * pos
            x1 = sx + ux * npos
            y1 = sy + uy * npos
            self.draw.line((x0, y0, x1, y1), fill=color, width=width)
            pos = npos + gap


def main() -> None:
    w, h = 2200, 1300
    img = Image.new("RGB", (w * SCALE, h * SCALE), Theme.BG)
    shadow_layer = Image.new("RGBA", (w * SCALE, h * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = ProRenderer(draw, shadow_layer)

    r.text((w / 2, 56), "SDX MODEL ARCHITECTURE", size=42, bold=True)
    r.text((w / 2, 98), "Code-aligned view from train.py, sample.py, models/, data/, diffusion/, utils/", size=18, color=Theme.TEXT_SUB)

    pixel_col = (40, 150, 430, 1110)
    latent_col = (460, 150, 1480, 1110)
    cond_col = (1510, 150, 2160, 1110)
    r.card(pixel_col, Theme.PIXEL[0], Theme.PIXEL[1], radius=20)
    r.card(latent_col, Theme.LATENT[0], Theme.LATENT[1], radius=20)
    r.card(cond_col, Theme.COND[0], Theme.COND[1], radius=20)
    r.text((235, 185), "PIXEL SPACE", size=24, bold=True, color=Theme.PIXEL[1])
    r.text((970, 185), "LATENT SPACE + DIT CORE", size=24, bold=True, color=Theme.LATENT[1])
    r.text((1835, 185), "CONDITIONING + AUXILIARY PATHS", size=24, bold=True, color=Theme.COND[1])

    r.card((70, 250, 400, 360), Theme.CARD_BG, Theme.PIXEL[1])
    r.text((235, 305), "Input image x\n(data/t2i_dataset.py)", size=16, bold=True)
    r.card((70, 390, 400, 520), Theme.CARD_BG, Theme.PIXEL[1])
    r.text((235, 455), "VAE / RAE Encoder\n(latent encode)", size=17, bold=True)
    r.card((70, 760, 400, 900), Theme.CARD_BG, Theme.PIXEL[1])
    r.text((235, 830), "VAE / RAE Decoder\n(latent -> pixel)", size=17, bold=True)
    r.card((70, 930, 400, 1060), Theme.CARD_BG, Theme.PIXEL[1])
    r.text((235, 995), "Output image x_hat\n(+ postprocess)", size=17, bold=True)

    r.card((500, 240, 1440, 330), Theme.CARD_BG, Theme.LATENT[1], shadow=False)
    r.text((970, 285), "Forward process: q(z_t | z0, t)  |  beta schedule + timestep sampling", size=19, bold=True)
    r.arrow((430, 455), (500, 455), color=Theme.PIXEL[1], width=3)
    r.text((485, 475), "z0", size=14, color=Theme.TEXT_SUB, align="left")

    r.card((500, 350, 1440, 770), Theme.CARD_BG, Theme.LATENT[1], radius=16)
    r.text((970, 385), "Denoising Transformer Core (models/dit_text.py)", size=22, bold=True)
    r.card((540, 430, 760, 720), Theme.CORE[0], Theme.CORE[1], shadow=False)
    r.text((650, 505), "Patch embed\n+ position\n+ timestep", size=16, bold=True)
    r.card((790, 430, 1120, 720), Theme.CORE[0], Theme.CORE[1], shadow=False)
    r.text((955, 505), "DiT blocks\nself-attn + cross-attn\nadaLN / FFN\n(optional MoE, AR blocks)", size=16, bold=True)
    r.card((1150, 430, 1400, 720), Theme.CORE[0], Theme.CORE[1], shadow=False)
    r.text((1275, 505), "Output head\nepsilon/v\n(+ flow velocity)", size=16, bold=True)

    r.card((500, 790, 1440, 910), Theme.INFER[0], Theme.INFER[1], radius=14)
    r.text((970, 850), "Sampling loop (sample.py + diffusion.sample_loop): CFG, scheduler/solver, optional rescale/threshold", size=18, bold=True)

    r.card((1540, 230, 2130, 340), Theme.CARD_BG, Theme.COND[1])
    r.text((1835, 285), "Prompt stack\nutils/prompt/content_controls + neg_filter + emphasis", size=16, bold=True)
    r.card((1540, 370, 2130, 510), Theme.CARD_BG, Theme.COND[1])
    r.text((1835, 440), "Text encoder bundle\nT5 (default) or triple: T5 + CLIP-L + CLIP-bigG fusion", size=16, bold=True)
    r.card((1540, 540, 2130, 680), Theme.CARD_BG, Theme.COND[1])
    r.text((1835, 610), "Adapters\nLoRA / DoRA / LyCORIS\n(role budgets + stage routing)", size=16, bold=True)
    r.card((1540, 710, 2130, 850), Theme.CARD_BG, Theme.COND[1])
    r.text((1835, 780), "Optional condition paths\ncontrol_image, reference tokens,\nstyle/creativity conditioning", size=16, bold=True)
    r.card((1540, 880, 2130, 1040), Theme.CARD_BG, Theme.NATIVE[1])
    r.text((1835, 960), "Native + tooling layer\nCUDA kernels, Rust tools, toolkit/env checks,\ncheckpoint manager + manifests/loggers", size=16, bold=True)

    r.arrow((1540, 440), (1400, 560), color=Theme.COND[1], width=3)
    r.arrow((1540, 610), (1120, 640), color=Theme.COND[1], width=3, dashed=True)
    r.arrow((1540, 780), (1400, 700), color=Theme.COND[1], width=3, dashed=True)

    r.card((40, 1140, 2160, 1260), Theme.CARD_BG, Theme.CORE[1], radius=16)
    r.text((1100, 1200), "Training objectives (train.py): main diffusion/flow loss + optional bridge/OT/part-aware losses; runtime bf16+compile+DDP; artifacts checkpoints/EMA/config/logs", size=17, bold=True)

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=6 * SCALE))
    out = Image.new("RGBA", (w * SCALE, h * SCALE), (0, 0, 0, 0))
    out.paste(shadow_layer, (0, 0), shadow_layer)
    out.alpha_composite(img.convert("RGBA"))
    out = out.resize((w, h), resample=Image.Resampling.LANCZOS)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    p1 = SAVE_PATH / FILE_NAME
    p2 = SAVE_PATH / ALT_FILE_NAME
    out.convert("RGB").save(p1, "PNG", optimize=True)
    out.convert("RGB").save(p2, "PNG", optimize=True)
    print(f"Saved:\n- {p1}\n- {p2}")


if __name__ == "__main__":
    main()
