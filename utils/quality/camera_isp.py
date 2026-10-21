"""Physically grounded capture finish: lens optics, then a digital sensor + ISP or a film emulsion.

The generated image is treated as a camera's display-referred output and unprocessed back to
linear raw as in Brooks et al., "Unprocessing Images for Learned Raw Denoising" (CVPR 2019):
inverse tone curve, sRGB EOTF, inverse colour matrix, inverse white-balance gains. Lens and
sensor effects are applied there and a forward ISP re-renders the image, so with every effect
off the round trip is the identity and only the capture signature is added.

Medium routing (``plan_finish``): photographs get the digital camera, film prompts the
emulsion model, and every other medium (anime, illustration, painting, 3D, pixel art) only the
dithered output conversion.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from utils.quality.quality import rgb_to_luminance

__all__ = [
    "CAMERA_PRESETS",
    "CameraProfile",
    "FilmStock",
    "Lens",
    "NO_LENS",
    "apply_diffusion_filter",
    "apply_finish",
    "apply_lens",
    "apply_lens_srgb",
    "demosaic_malvar",
    "film_stock_for_prompt",
    "jpeg_roundtrip",
    "lateral_chromatic_aberration",
    "lab_to_srgb",
    "linear_to_srgb",
    "mosaic_rggb",
    "plan_finish",
    "poisson_gaussian_noise",
    "save_rgb",
    "simulate_capture",
    "simulate_film",
    "srgb_to_lab",
    "srgb_to_linear",
    "to_uint8",
    "user_lens",
    "vignette_gain",
]


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB EOTF on values in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Inverse of ``srgb_to_linear`` (sRGB OETF), clipped to [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``[..., 3]`` in [0, 1] -> CIE Lab (D65)."""
    lin = srgb_to_linear(rgb)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Inverse of ``srgb_to_lab``, clipped to [0, 1]."""
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps = 216 / 24389
    kappa = 24389 / 27
    xr = np.where(fx**3 > eps, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(l > kappa * eps, fy**3, l / kappa)
    zr = np.where(fz**3 > eps, fz**3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=-1)
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=np.float64,
    )
    return linear_to_srgb(xyz @ m_inv.T)


def _unit(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32)[..., :3] / 255.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Lens:
    """Residual lens character after in-camera corrections."""

    half_fov_deg: float = 23.4  # half the diagonal field of view (50 mm on full frame)
    vignette: float = 0.5  # share of the cos^4 falloff left after lens-shading correction
    ca_red: float = 0.0008  # lateral CA: red magnification - 1 at the frame corner (green = reference)
    ca_blue: float = -0.0006
    psf_sigma: float = 0.45  # optical blur, Gaussian PSF sigma in pixels


NO_LENS = Lens(vignette=0.0, ca_red=0.0, ca_blue=0.0, psf_sigma=0.0)


def user_lens(base: Lens, vignette: float = 0.0, chromatic_aberration: float = 0.0) -> Lens:
    """Apply the 0-1 ``--vignette`` / ``--chromatic-aberration`` knobs to ``base`` (0 keeps its value).

    Chromatic aberration 1.0 is a 0.4% red magnification at the corner (about 3 px on a 1024 px frame).
    """
    if vignette > 0.0:
        base = replace(base, vignette=float(min(1.0, vignette)))
    if chromatic_aberration > 0.0:
        k = 0.004 * float(min(1.0, chromatic_aberration))
        base = replace(base, ca_red=k, ca_blue=-0.8 * k)
    return base


def _radial(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets from the optical centre and squared radius normalised to 1 at the corner."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    return dy, dx, (dx * dx + dy * dy) / max(cx * cx + cy * cy, 1e-6)


def vignette_gain(h: int, w: int, lens: Lens) -> np.ndarray:
    """Natural vignetting (cos^4 law), scaled by the share left after lens-shading correction."""
    t2 = _radial(h, w)[2] * np.float32(np.tan(np.radians(lens.half_fov_deg)) ** 2)
    cos4 = 1.0 / (1.0 + t2) ** 2
    return (1.0 - lens.vignette * (1.0 - cos4)).astype(np.float32)


def lateral_chromatic_aberration(lin: np.ndarray, ca_red: float, ca_blue: float) -> np.ndarray:
    """Lateral CA as per-channel radial magnification ``1 + k r^2`` about the optical centre.

    Cubic-spline resampling at exact (unquantised) sub-pixel positions; green is the reference.
    """
    if ca_red == 0.0 and ca_blue == 0.0:
        return lin
    from scipy.ndimage import map_coordinates

    h, w = lin.shape[:2]
    dy, dx, r2 = _radial(h, w)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    out = lin.copy()
    for c, k in ((0, ca_red), (2, ca_blue)):
        if k == 0.0:
            continue
        s = 1.0 + np.float32(k) * r2
        coords = np.stack([cy + dy / s, cx + dx / s])
        out[..., c] = map_coordinates(lin[..., c], coords, order=3, mode="mirror")
    return out


def apply_lens(lin: np.ndarray, lens: Lens) -> np.ndarray:
    """Vignette, lateral CA and optical blur on a linear-light (H, W, 3) image."""
    h, w = lin.shape[:2]
    out = lin * vignette_gain(h, w, lens)[..., None]
    out = lateral_chromatic_aberration(out, lens.ca_red, lens.ca_blue)
    if lens.psf_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), lens.psf_sigma, borderType=cv2.BORDER_REFLECT_101)
    return out


def apply_lens_srgb(image: np.ndarray, lens: Lens) -> np.ndarray:
    """``apply_lens`` for an sRGB [0, 255] image (decoded to linear light and back)."""
    lin = apply_lens(srgb_to_linear(_unit(image)), lens)
    return (linear_to_srgb(lin) * 255.0).astype(np.float32)


def apply_diffusion_filter(image: np.ndarray, amount: float, radius_frac: float = 0.02) -> np.ndarray:
    """Mist / soft-focus filter: a share ``amount`` of the light is scattered into a wide Gaussian PSF.

    Done in linear light, so highlights bloom and shadows pick up veiling glare, as with a
    real Pro-Mist or diffusion filter. ``radius_frac`` is the PSF sigma as a share of the diagonal.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 0.0:
        return image
    lin = srgb_to_linear(_unit(image))
    h, w = lin.shape[:2]
    sigma = max(0.5, float(radius_frac) * float(np.hypot(h, w)))
    glow = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    out = (linear_to_srgb((1.0 - a) * lin + a * glow) * 255.0).astype(np.float32)
    return to_uint8(out) if image.dtype == np.uint8 else out


def poisson_gaussian_noise(
    raw: np.ndarray, *, full_well_e: float, read_noise_e: float, rng: np.random.Generator
) -> np.ndarray:
    """Heteroscedastic sensor noise (Foi et al., IEEE TIP 2008): Poisson shot noise plus Gaussian read noise.

    ``raw`` is linear signal normalised to the white level, so
    ``var = raw / full_well_e + (read_noise_e / full_well_e) ** 2``.
    """
    electrons = rng.poisson(np.clip(raw, 0.0, None).astype(np.float64) * full_well_e)
    noisy = electrons + rng.normal(0.0, read_noise_e, raw.shape)
    return (noisy / full_well_e).astype(np.float32)


def mosaic_rggb(rgb: np.ndarray) -> np.ndarray:
    """Sample an (H, W, 3) image through an RGGB Bayer colour filter array."""
    cfa = np.array(rgb[..., 1], dtype=np.float32)
    cfa[0::2, 0::2] = rgb[0::2, 0::2, 0]
    cfa[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return cfa


# Malvar, He & Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned
# color images" (ICASSP 2004): 5x5 gradient-corrected bilinear kernels.
_MHC_G = (
    np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32)
    / 8.0
)
_MHC_ROW = (
    np.array(
        [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]], np.float32
    )
    / 8.0
)
_MHC_COL = np.ascontiguousarray(_MHC_ROW.T)
_MHC_DIAG = (
    np.array(
        [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32
    )
    / 8.0
)


def demosaic_malvar(cfa: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler demosaic of an RGGB mosaic to (H, W, 3)."""
    cfa = np.ascontiguousarray(cfa, dtype=np.float32)

    def filt(k: np.ndarray) -> np.ndarray:
        # Reflect-101 keeps the Bayer phase at the borders.
        return cv2.filter2D(cfa, -1, k, borderType=cv2.BORDER_REFLECT_101)

    g_rb, row, col, diag = filt(_MHC_G), filt(_MHC_ROW), filt(_MHC_COL), filt(_MHC_DIAG)
    r, g, b = np.empty_like(cfa), g_rb, np.empty_like(cfa)
    r[0::2, 0::2], b[0::2, 0::2] = cfa[0::2, 0::2], diag[0::2, 0::2]
    r[0::2, 1::2], g[0::2, 1::2], b[0::2, 1::2] = row[0::2, 1::2], cfa[0::2, 1::2], col[0::2, 1::2]
    r[1::2, 0::2], g[1::2, 0::2], b[1::2, 0::2] = col[1::2, 0::2], cfa[1::2, 0::2], row[1::2, 0::2]
    r[1::2, 1::2], b[1::2, 1::2] = diag[1::2, 1::2], cfa[1::2, 1::2]
    return np.stack([r, g, b], axis=-1)


@dataclass(frozen=True, slots=True)
class CameraProfile:
    """Digital camera: lens, sensor, colour pipeline and JPEG engine.

    ``ccm`` maps white-balanced camera RGB to linear sRGB (rows sum to 1, so white is kept); the
    matrices below are representative daylight matrices, not measurements of a specific body.
    Off-diagonals are kept non-positive so the inverse is non-negative and unprocessed raw stays >= 0.
    """

    name: str
    lens: Lens = field(default_factory=Lens)
    iso: int = 100
    base_iso: int = 100
    full_well_e: float = 30000.0  # electrons at base ISO
    read_noise_e: float = 2.5  # pre-amplifier read noise (electrons)
    post_read_noise_e: float = 10.0  # post-amplifier read noise in base-ISO electrons (falls as 1/gain)
    wb_gains: tuple[float, float, float] = (2.0, 1.0, 1.55)
    ccm: tuple[tuple[float, float, float], ...] = ((1.80, -0.66, -0.14), (-0.20, 1.55, -0.35), (-0.02, -0.50, 1.52))
    tone: float = 0.5  # share of smoothstep in the S-shaped tone curve
    cfa: bool = True
    chroma_nr: float = 0.0  # Gaussian sigma on Cb/Cr, pixels
    sharpen: float = 0.5
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 1.0 / 255.0
    jpeg_quality: int = 92


CAMERA_PRESETS: dict[str, CameraProfile] = {
    "phone": CameraProfile(
        "phone",
        lens=Lens(half_fov_deg=38.0, vignette=0.25, ca_red=0.0006, ca_blue=-0.0005, psf_sigma=0.6),
        iso=50,
        base_iso=50,
        full_well_e=15000.0,  # effective, after multi-frame merging
        read_noise_e=1.5,
        post_read_noise_e=3.0,
        wb_gains=(1.9, 1.0, 1.7),
        ccm=((1.65, -0.50, -0.15), (-0.28, 1.58, -0.30), (-0.02, -0.60, 1.62)),
        tone=0.65,
        chroma_nr=2.0,
        sharpen=0.8,
        jpeg_quality=88,
    ),
    "dslr": CameraProfile(
        "dslr",
        lens=Lens(half_fov_deg=23.4, vignette=0.6, ca_red=0.0010, ca_blue=-0.0008, psf_sigma=0.45),
        full_well_e=40000.0,
        read_noise_e=2.5,
        post_read_noise_e=12.0,
        tone=0.45,
        chroma_nr=0.7,
        sharpen=0.4,
        sharpen_radius=0.9,
        jpeg_quality=95,
    ),
    "compact": CameraProfile(
        "compact",
        lens=Lens(half_fov_deg=31.0, vignette=0.4, ca_red=0.0015, ca_blue=-0.0012, psf_sigma=0.65),
        full_well_e=12000.0,
        read_noise_e=3.0,
        post_read_noise_e=8.0,
        wb_gains=(1.95, 1.0, 1.6),
        ccm=((1.70, -0.55, -0.15), (-0.25, 1.50, -0.25), (-0.05, -0.55, 1.60)),
        tone=0.55,
        chroma_nr=1.0,
        sharpen=0.6,
        jpeg_quality=90,
    ),
}


def _tone(x: np.ndarray, t: float) -> np.ndarray:
    """Camera tone curve: blend of identity and smoothstep (Brooks et al. use the full smoothstep)."""
    return (1.0 - t) * x + t * x * x * (3.0 - 2.0 * x)


def _inverse_tone(y: np.ndarray, t: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    return np.interp(y, _tone(grid, t), grid).astype(np.float32)


def _chroma_denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Blur Cb/Cr (BT.601 colour differences) and keep luma, as camera chroma NR does."""
    if sigma <= 0.0:
        return rgb
    y = rgb_to_luminance(rgb)
    cr = cv2.GaussianBlur(rgb[..., 0] - y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    cb = cv2.GaussianBlur(rgb[..., 2] - y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    r, b = y + cr, y + cb
    return np.stack([r, (y - 0.299 * r - 0.114 * b) / 0.587, b], axis=-1)


def _sharpen_luma(rgb: np.ndarray, amount: float, radius: float, threshold: float) -> np.ndarray:
    """In-camera unsharp mask on luma with coring, so flat-area noise is not amplified."""
    if amount <= 0.0:
        return rgb
    y = rgb_to_luminance(rgb)
    d = y - cv2.GaussianBlur(y, (0, 0), radius, borderType=cv2.BORDER_REFLECT_101)
    d = np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
    return rgb + amount * d[..., None]


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode as baseline JPEG with 4:2:0 chroma subsampling; returns uint8."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(to_uint8(image)).save(buf, "JPEG", quality=int(quality), subsampling=2)
    with Image.open(io.BytesIO(buf.getvalue())) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def simulate_capture(image: np.ndarray, profile: CameraProfile, *, seed: int = 0, jpeg: bool = True) -> np.ndarray:
    """Re-capture an sRGB [0, 255] image through ``profile``.

    Returns uint8 when the JPEG stage runs, else float32 [0, 255].
    """
    rng = np.random.default_rng(int(seed))
    ccm = np.asarray(profile.ccm, dtype=np.float32)
    gains = np.asarray(profile.wb_gains, dtype=np.float32)

    lin = srgb_to_linear(_inverse_tone(_unit(image), profile.tone))
    raw = apply_lens((lin @ np.linalg.inv(ccm).T) / gains, profile.lens)

    gain = max(profile.iso, profile.base_iso) / profile.base_iso
    full_well = profile.full_well_e / gain
    read = float(np.hypot(profile.read_noise_e, profile.post_read_noise_e / gain))
    if profile.cfa:
        cfa = poisson_gaussian_noise(mosaic_rggb(raw), full_well_e=full_well, read_noise_e=read, rng=rng)
        # White balance on the mosaic, then clip at the white level (no magenta highlights).
        cfa = np.clip(cfa * mosaic_rggb(np.broadcast_to(gains, raw.shape)), 0.0, 1.0)
        cam = demosaic_malvar(cfa)
    else:
        cam = np.clip(poisson_gaussian_noise(raw, full_well_e=full_well, read_noise_e=read, rng=rng) * gains, 0.0, 1.0)

    out = _tone(linear_to_srgb(cam @ ccm.T), profile.tone)
    out = _chroma_denoise(out, profile.chroma_nr)
    out = _sharpen_luma(out, profile.sharpen, profile.sharpen_radius, profile.sharpen_threshold)
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.float32)
    if jpeg and profile.jpeg_quality > 0:
        return jpeg_roundtrip(out, profile.jpeg_quality)
    return out


@dataclass(frozen=True, slots=True)
class FilmStock:
    """Colour-negative or B&W emulsion, scanned.

    ``granularity`` is diffuse RMS granularity x1000 at net density 1.0 through a 48 um aperture
    (the datasheet figure; Tri-X 400 is 17). 0 derives it from ISO: colour negative ~4*sqrt(ISO/100),
    conventional B&W ~8.5*sqrt(ISO/100).
    """

    name: str
    iso: int = 400
    monochrome: bool = False
    granularity: float = 0.0
    frame_width_mm: float = 36.0
    clump_um: float = 12.0  # grain / dye-cloud clump size
    gamma: float = 0.65  # negative contrast on the straight-line part of the H&D curve
    lens: Lens = field(default_factory=lambda: Lens(vignette=0.6, psf_sigma=0.5))

    def rms_granularity(self) -> float:
        if self.granularity > 0.0:
            return self.granularity
        return (8.5 if self.monochrome else 4.0) * float(np.sqrt(max(self.iso, 6) / 100.0))


_D_RANGE = 2.2  # usable density range of the negative above base + fog
_LOG_E_MID = float(np.log10(0.18)) + 0.6  # H&D inflection two stops over mid grey


def _hd_curve(log_e: np.ndarray, gamma: float) -> np.ndarray:
    """Net negative density vs log10 exposure: logistic H&D curve with mid-slope ``gamma``."""
    k = 4.0 * gamma / _D_RANGE
    return _D_RANGE / (1.0 + np.exp(-k * np.log(10.0) * (log_e - _LOG_E_MID)))


def _hd_inverse(density: np.ndarray, gamma: float) -> np.ndarray:
    k = 4.0 * gamma / _D_RANGE
    s = np.clip(density / _D_RANGE, 1e-6, 1.0 - 1e-6)
    return _LOG_E_MID - np.log(1.0 / s - 1.0) / (k * np.log(10.0))


def simulate_film(image: np.ndarray, stock: FilmStock, *, seed: int = 0) -> np.ndarray:
    """Expose an sRGB [0, 255] image onto ``stock`` and scan it back; returns float32 [0, 255].

    Grain lives in density: each dye layer (one for B&W) gets an independent field whose RMS
    follows Selwyn's law (sigma_D * sqrt(area) constant) scaled to the scan pixel, grows with
    sqrt(net density) (Nutting), and is spatially correlated over the clump size. Density is then
    mapped back through the same H&D curve, so tone is unchanged and only grain is added.
    """
    lin = apply_lens(srgb_to_linear(_unit(image)), stock.lens)
    h, w = lin.shape[:2]
    pitch_um = stock.frame_width_mm * 1000.0 / max(h, w)
    sigma_d = stock.rms_granularity() / 1000.0 * 48.0 / max(pitch_um, stock.clump_um)

    layers = rgb_to_luminance(lin)[..., None] if stock.monochrome else lin
    exposure = np.clip(layers, 1e-4, None)
    density = _hd_curve(np.log10(exposure), stock.gamma)

    rng = np.random.default_rng(int(seed))
    field_ = rng.standard_normal(density.shape).astype(np.float32)
    blur = stock.clump_um / pitch_um
    if blur >= 0.3:
        field_ = cv2.GaussianBlur(field_, (0, 0), blur, borderType=cv2.BORDER_REFLECT_101).reshape(density.shape)
        field_ /= field_.reshape(-1, density.shape[-1]).std(axis=0) + 1e-8
    grainy = density + sigma_d * np.sqrt(np.maximum(density, 0.0)) * field_
    ratio = np.power(10.0, _hd_inverse(grainy, stock.gamma)) / exposure
    return (linear_to_srgb(lin * ratio) * 255.0).astype(np.float32)


_FILM_RE = re.compile(
    r"\b(?:analog(?:ue)? (?:film|photo\w*)|film (?:photo\w*|grain|still|camera|stock|scan|look)|"
    r"(?:35|120|16|8) ?mm film|shot on (?:35|120|16|8) ?mm|super ?8|medium format film|"
    r"black[- ]and[- ]white film|b&w film|monochrome film|expired film|disposable camera|"
    r"kodak|portra|ektar|ektachrome|kodachrome|tri-?x|t-?max|ilford|hp5|cinestill|vision ?3|"
    r"fuji(?:film)? (?:superia|velvia|provia|pro ?400h|c200)|polaroid|instax|lomograph\w*)\b",
    re.I,
)
_MONO_RE = re.compile(
    r"\b(?:black[- ]and[- ]white|b&w|monochrome|tri-?x|t-?max|ilford|hp5|noir|greyscale|grayscale)\b", re.I
)
_ISO_RE = re.compile(r"\biso[ -]?(\d{2,5})\b", re.I)
_PHONE_RE = re.compile(
    r"\b(?:iphone|smartphone|phone (?:photo|camera|pic|snapshot)|camera phone|selfie|android phone|"
    r"pixel \d|galaxy s\d+|snapchat|instagram story)\b",
    re.I,
)
_COMPACT_RE = re.compile(
    r"\b(?:point[- ]and[- ]shoot|compact (?:digital )?camera|digicam|ccd (?:camera|sensor)|"
    r"early digital camera|2000s digital (?:camera|photo)|powershot|cybershot|coolpix)\b",
    re.I,
)
_LOW_LIGHT_RE = re.compile(
    r"\b(?:night\w*|midnight|low[- ]light|dim(?:ly)?[- ]lit|candle\w*|concert|nightclub|moonlight|"
    r"starlight|blue hour|dark (?:room|bar|alley|street))\b",
    re.I,
)
_INDOOR_RE = re.compile(
    r"\b(?:indoors?|interior|dusk|twilight|overcast|rainy|restaurant|cafe|bar|office|bedroom|kitchen)\b", re.I
)

# --photo-grain-style -> (iso, frame width mm); clean_digital stays on the digital camera.
_GRAIN_STYLE_FILM = {"fine_35mm": (100, 36.0), "medium_35mm": (400, 36.0), "heavy_16mm": (500, 10.26)}


def _explicit_iso(prompt: str) -> int:
    m = _ISO_RE.search(prompt or "")
    return int(np.clip(int(m.group(1)), 25, 102400)) if m else 0


def film_stock_for_prompt(prompt: str, grain_style: str = "none", iso: int = 0) -> FilmStock:
    """Pick an emulsion and frame size from film-stock cues in the prompt."""
    p = (prompt or "").lower()
    mono = bool(_MONO_RE.search(p))
    width, clump = 36.0, 12.0
    if re.search(r"\b16 ?mm\b", p):
        stock_iso, width = 500, 10.26
    elif re.search(r"\b(?:super ?8|8 ?mm)\b", p):
        stock_iso, width = 200, 5.79
    elif re.search(r"\b(?:polaroid|instax)\b", p):
        stock_iso, width, clump = 640, 79.0, 15.0
    elif re.search(r"\b(?:120 film|medium format)\b", p):
        stock_iso, width = 400, 56.0
    elif "ektar" in p:
        stock_iso = 100
    elif re.search(r"\bt-?max\b", p):
        stock_iso = 100
    elif re.search(r"\b(?:cinestill|vision ?3|800t|500t)\b", p):
        stock_iso = 800
    else:
        stock_iso = 400
    if grain_style in _GRAIN_STYLE_FILM:
        stock_iso, width = _GRAIN_STYLE_FILM[grain_style]
    stock_iso = int(iso or _explicit_iso(p) or stock_iso)
    granularity = 17.0 if re.search(r"\b(?:tri-?x|hp5)\b", p) else 0.0
    return FilmStock(
        f"{'bw' if mono else 'color'}_{stock_iso}",
        iso=stock_iso,
        monochrome=mono,
        granularity=granularity,
        frame_width_mm=width,
        clump_um=clump,
    )


def _non_photographic(prompt: str) -> bool:
    from config.defaults.art_mediums import detect_all_mediums, detect_medium_ids

    families = {m.family for m in detect_all_mediums(prompt)}
    return bool(families & {"2d", "traditional", "3d"}) or bool(detect_medium_ids(prompt, include_photography=False))


def plan_finish(
    prompt: str,
    *,
    camera: str = "auto",
    iso: int = 0,
    grain_style: str = "none",
    human_media: str = "none",
    vignette: float = 0.0,
    chromatic_aberration: float = 0.0,
) -> CameraProfile | FilmStock | None:
    """Route by medium: photo -> digital camera, film -> emulsion, anything else -> None (no capture finish).

    ``camera`` forces a preset (``phone``/``dslr``/``compact``/``film``); ``none`` disables.
    ``vignette`` / ``chromatic_aberration`` are the explicit 0-1 knobs (see ``user_lens``).
    """
    cam = str(camera or "auto").lower().strip()
    style = str(grain_style or "none").lower().strip()
    p = prompt or ""
    if cam in ("none", "off"):
        return None
    if cam == "auto":
        if _non_photographic(p):
            return None
        if style in _GRAIN_STYLE_FILM or str(human_media).lower() == "film" or _FILM_RE.search(p):
            cam = "film"
        elif str(human_media).lower() == "dslr":
            cam = "dslr"
        elif _PHONE_RE.search(p):
            cam = "phone"
        elif _COMPACT_RE.search(p):
            cam = "compact"
        else:
            cam = "dslr"
    if cam == "film":
        plan: CameraProfile | FilmStock = film_stock_for_prompt(p, style, iso)
    else:
        profile = CAMERA_PRESETS.get(cam, CAMERA_PRESETS["dslr"])
        chosen = iso or _explicit_iso(p)
        if not chosen:
            boost = 16 if _LOW_LIGHT_RE.search(p) else 4 if _INDOOR_RE.search(p) else 1
            chosen = profile.base_iso * boost
        plan = replace(profile, iso=int(chosen))
    return replace(plan, lens=user_lens(plan.lens, vignette, chromatic_aberration))


def apply_finish(
    image: np.ndarray,
    plan: CameraProfile | FilmStock | None,
    *,
    seed: int = 0,
    jpeg_quality: int = 0,
    jpeg: bool = True,
) -> np.ndarray:
    """Run a ``plan_finish`` result on an sRGB [0, 255] image (``None`` returns it unchanged)."""
    if isinstance(plan, FilmStock):
        return simulate_film(image, plan, seed=seed)
    if isinstance(plan, CameraProfile):
        if jpeg_quality > 0:
            plan = replace(plan, jpeg_quality=int(np.clip(jpeg_quality, 1, 100)))
        return simulate_capture(image, plan, seed=seed, jpeg=jpeg)
    return image


def to_uint8(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Quantise [0, 255] to uint8 once; with ``rng``, TPDF dither (two uniform LSB) to avoid banding."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    if rng is not None:
        x = x + (rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def save_rgb(image: np.ndarray, path: str | Path, *, sixteen_bit: bool = False, seed: int = 0) -> np.ndarray:
    """Save an RGB [0, 255] image; float input is dithered to 8 bit or written as 16-bit PNG/TIFF.

    Returns the 8-bit image (for grids and previews).
    """
    path = Path(path)
    u8 = to_uint8(image, np.random.default_rng(int(seed)))
    if sixteen_bit and path.suffix.lower() in (".png", ".tif", ".tiff"):
        u16 = np.clip(np.rint(np.asarray(image, dtype=np.float32) * (65535.0 / 255.0)), 0, 65535).astype(np.uint16)
        ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(u16, cv2.COLOR_RGB2BGR))
        if ok:
            path.write_bytes(buf.tobytes())
            return u8
    from PIL import Image

    Image.fromarray(u8).save(path)
    return u8
