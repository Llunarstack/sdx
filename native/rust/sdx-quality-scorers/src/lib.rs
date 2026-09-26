//! Image quality heuristics for human-likeness / test-time pick — C ABI.
//!
//! Exports Laplacian variance (sharpness), highlight clip fraction, and
//! midtone histogram entropy on RGB u8 HWC buffers.

use std::slice;

fn gray_at(rgb: &[u8], _h: usize, w: usize, y: usize, x: usize) -> f32 {
    let i = (y * w + x) * 3;
    let r = rgb[i] as f32;
    let g = rgb[i + 1] as f32;
    let b = rgb[i + 2] as f32;
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Laplacian variance of luma (higher = sharper). Returns -1 on error.
#[no_mangle]
pub unsafe extern "C" fn sdx_quality_laplacian_var_u8(rgb: *const u8, h: usize, w: usize) -> f64 {
    if rgb.is_null() || h < 3 || w < 3 {
        return -1.0;
    }
    let n = h * w * 3;
    let src = slice::from_raw_parts(rgb, n);
    laplacian_var(src, h, w)
}

/// Fraction of pixels with luma >= `thr` (0–255). Returns -1 on error.
#[no_mangle]
pub unsafe extern "C" fn sdx_quality_highlight_frac_u8(rgb: *const u8, h: usize, w: usize, thr: f32) -> f64 {
    if rgb.is_null() || h == 0 || w == 0 {
        return -1.0;
    }
    let src = slice::from_raw_parts(rgb, h * w * 3);
    highlight_frac(src, h, w, thr)
}

/// Shannon entropy (bits) of midtone luma histogram (luma in [lo, hi]). Returns -1 on error.
#[no_mangle]
pub unsafe extern "C" fn sdx_quality_midtone_entropy_u8(
    rgb: *const u8,
    h: usize,
    w: usize,
    lo: f32,
    hi: f32,
) -> f64 {
    if rgb.is_null() || h == 0 || w == 0 {
        return -1.0;
    }
    let src = slice::from_raw_parts(rgb, h * w * 3);
    midtone_entropy(src, h, w, lo, hi)
}

pub fn laplacian_var(rgb: &[u8], h: usize, w: usize) -> f64 {
    let mut vals: Vec<f64> = Vec::with_capacity((h - 2) * (w - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let c = gray_at(rgb, h, w, y, x) as f64;
            let lap = -4.0 * c
                + gray_at(rgb, h, w, y - 1, x) as f64
                + gray_at(rgb, h, w, y + 1, x) as f64
                + gray_at(rgb, h, w, y, x - 1) as f64
                + gray_at(rgb, h, w, y, x + 1) as f64;
            vals.push(lap);
        }
    }
    if vals.is_empty() {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
    var
}

pub fn highlight_frac(rgb: &[u8], h: usize, w: usize, thr: f32) -> f64 {
    let t = thr.clamp(0.0, 255.0);
    let mut hit = 0usize;
    let n = h * w;
    for y in 0..h {
        for x in 0..w {
            if gray_at(rgb, h, w, y, x) >= t {
                hit += 1;
            }
        }
    }
    hit as f64 / n.max(1) as f64
}

pub fn midtone_entropy(rgb: &[u8], h: usize, w: usize, lo: f32, hi: f32) -> f64 {
    let mut hist = [0u64; 64];
    let lo = lo.clamp(0.0, 255.0);
    let hi = hi.clamp(lo + 1.0, 255.0);
    let mut total = 0u64;
    for y in 0..h {
        for x in 0..w {
            let g = gray_at(rgb, h, w, y, x);
            if g < lo || g > hi {
                continue;
            }
            let t = ((g - lo) / (hi - lo) * 63.0).round() as usize;
            hist[t.min(63)] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let mut ent = 0.0f64;
    let tot = total as f64;
    for &c in &hist {
        if c == 0 {
            continue;
        }
        let p = c as f64 / tot;
        ent -= p * p.log2();
    }
    ent
}
