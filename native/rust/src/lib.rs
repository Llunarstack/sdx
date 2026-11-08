//! C ABI kernels for Python ctypes (`sdx_native.ffi`): VP-diffusion math, prompt token Jaccard,
//! highlight-clip fraction and foreground blob counting.
//!
//! # Safety
//! Every exported function takes raw pointers with explicit lengths; callers must pass buffers valid
//! for the stated element counts (outputs must not overlap inputs).

use std::collections::HashSet;
use std::f64::consts::FRAC_PI_2;
use std::slice;
use std::str;

// ---------------------------------------------------------------------------------------------
// VP-diffusion math (mirrors diffusion/schedules.py). Return 0 on success, -2 on bad arguments.
// ---------------------------------------------------------------------------------------------

/// `out[t] = prod_{s<=t} (1 - betas[s])`.
#[no_mangle]
pub unsafe extern "C" fn sdx_alpha_cumprod_f64(betas: *const f64, out: *mut f64, n: usize) -> i32 {
    if betas.is_null() || out.is_null() || n == 0 {
        return -2;
    }
    let mut acc = 1.0f64;
    for (o, &beta) in slice::from_raw_parts_mut(out, n).iter_mut().zip(slice::from_raw_parts(betas, n)) {
        acc *= 1.0 - beta;
        *o = acc;
    }
    0
}

/// `out[t] = ac[t] / (1 - ac[t] + 1e-8)`.
#[no_mangle]
pub unsafe extern "C" fn sdx_snr_from_alpha_cumprod_f64(alpha_cumprod: *const f64, out: *mut f64, n: usize) -> i32 {
    if alpha_cumprod.is_null() || out.is_null() || n == 0 {
        return -2;
    }
    for (o, &a) in slice::from_raw_parts_mut(out, n).iter_mut().zip(slice::from_raw_parts(alpha_cumprod, n)) {
        *o = a / (1.0 - a + 1e-8);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sdx_linear_beta_schedule_f64(out: *mut f64, n: usize, beta_start: f64, beta_end: f64) -> i32 {
    if out.is_null() || n == 0 {
        return -2;
    }
    let o = slice::from_raw_parts_mut(out, n);
    if n == 1 {
        o[0] = beta_start;
        return 0;
    }
    let step = (beta_end - beta_start) / (n as f64 - 1.0);
    for (i, v) in o.iter_mut().enumerate() {
        *v = (beta_start + step * i as f64).clamp(1e-4, 0.999);
    }
    0
}

/// Squared-cosine v2: `alpha_bar(t) = cos^2((t/T + 0.008) / 1.008 * pi/2)`.
#[no_mangle]
pub unsafe extern "C" fn sdx_squaredcos_beta_schedule_v2_f64(out: *mut f64, n: usize, max_beta: f64) -> i32 {
    if out.is_null() || n == 0 {
        return -2;
    }
    let nf = n as f64;
    let ab = |i: usize| ((i as f64 / nf + 0.008) / 1.008 * FRAC_PI_2).cos().powi(2);
    for (i, o) in slice::from_raw_parts_mut(out, n).iter_mut().enumerate() {
        let beta = (1.0 - ab(i + 1) / ab(i).max(1e-12)).min(max_beta);
        *o = beta.clamp(1e-4, 0.999);
    }
    0
}

/// Cosine schedule (Nichol & Dhariwal, s = 0.01).
#[no_mangle]
pub unsafe extern "C" fn sdx_cosine_beta_schedule_f64(out: *mut f64, n: usize) -> i32 {
    if out.is_null() || n == 0 {
        return -2;
    }
    let nf = n as f64;
    let ab = |i: usize| ((i as f64 / nf + 0.01) / 1.01 * FRAC_PI_2).cos().powi(2);
    let ab0 = ab(0);
    for (i, o) in slice::from_raw_parts_mut(out, n).iter_mut().enumerate() {
        let a1 = ab(i) / ab0;
        let a2 = ab(i + 1) / ab0;
        *o = (1.0 - a2 / a1.max(1e-12)).clamp(1e-4, 0.999);
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Prompt token Jaccard (utils.prompt.style_native.text_overlap).
// ---------------------------------------------------------------------------------------------

fn token_set_lower(text: &str) -> HashSet<String> {
    text.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

/// Jaccard of comma/whitespace token sets in basis points (0-10000); 0 if either side is empty.
pub fn token_jaccard_basis_points(a: &str, b: &str) -> i32 {
    let sa = token_set_lower(a);
    let sb = token_set_lower(b);
    if sa.is_empty() || sb.is_empty() {
        return 0;
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.len() + sb.len() - inter;
    ((inter as f64 / union as f64) * 10000.0).round() as i32
}

/// Returns basis points, -2 on null pointers, -3 on invalid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn sdx_token_jaccard_utf8(a: *const u8, a_len: usize, b: *const u8, b_len: usize) -> i32 {
    if a.is_null() || b.is_null() {
        return -2;
    }
    match (
        str::from_utf8(slice::from_raw_parts(a, a_len)),
        str::from_utf8(slice::from_raw_parts(b, b_len)),
    ) {
        (Ok(a_s), Ok(b_s)) => token_jaccard_basis_points(a_s, b_s),
        _ => -3,
    }
}

// ---------------------------------------------------------------------------------------------
// Image heuristics.
// ---------------------------------------------------------------------------------------------

/// Fraction of RGB u8 HWC pixels whose luma (0.299/0.587/0.114) is >= `thr`; -1 on bad arguments.
#[no_mangle]
pub unsafe extern "C" fn sdx_quality_highlight_frac_u8(rgb: *const u8, h: usize, w: usize, thr: f32) -> f64 {
    if rgb.is_null() || h == 0 || w == 0 {
        return -1.0;
    }
    highlight_frac(slice::from_raw_parts(rgb, h * w * 3), thr)
}

pub fn highlight_frac(rgb: &[u8], thr: f32) -> f64 {
    let t = thr.clamp(0.0, 255.0);
    let n = rgb.len() / 3;
    let hit = rgb
        .chunks_exact(3)
        .filter(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32 >= t)
        .count();
    hit as f64 / n.max(1) as f64
}

/// Otsu threshold in [0, 1] from a 256-bin histogram, placed between bins.
fn otsu_threshold(gray: &[f32]) -> f32 {
    let mut hist = [0u64; 256];
    for &v in gray {
        hist[((v * 255.0).round() as i32).clamp(0, 255) as usize] += 1;
    }
    let total = gray.len() as f64;
    let sum_all: f64 = hist.iter().enumerate().map(|(t, &h)| t as f64 * h as f64).sum();
    let (mut w_b, mut sum_b, mut best_t, mut best_var) = (0.0f64, 0.0f64, 0usize, -1.0f64);
    for (t, &h) in hist.iter().enumerate() {
        w_b += h as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * h as f64;
        let d = sum_b / w_b - (sum_all - sum_b) / w_f;
        let var = w_b * w_f * d * d;
        if var > best_var {
            best_var = var;
            best_t = t;
        }
    }
    (best_t as f32 + 0.5) / 255.0
}

/// Count foreground blobs in a row-major `h*w` grayscale image (`f32` in [0, 1]); -1 on bad arguments.
///
/// Otsu threshold, background = class dominating the border ring, 4-connected components. Components
/// smaller than `min_area_frac` of the image, larger than 60% of it, or touching the border are not
/// counted. Mirrors `scripts/tools/evals.py::_count_blobs_numpy`.
#[no_mangle]
pub unsafe extern "C" fn sdx_count_blobs_f32(gray: *const f32, h: usize, w: usize, min_area_frac: f32) -> i64 {
    if gray.is_null() || h == 0 || w == 0 {
        return -1;
    }
    count_blobs(slice::from_raw_parts(gray, h * w), h, w, min_area_frac)
}

pub fn count_blobs(g: &[f32], h: usize, w: usize, min_area_frac: f32) -> i64 {
    let n = h * w;
    if n == 0 || g.len() < n {
        return -1;
    }
    let thr = otsu_threshold(&g[..n]);
    let dark: Vec<bool> = g[..n].iter().map(|&v| v < thr).collect();

    let mut border_dark = 0usize;
    for x in 0..w {
        border_dark += dark[x] as usize + dark[(h - 1) * w + x] as usize;
    }
    for y in 0..h {
        border_dark += dark[y * w] as usize + dark[y * w + w - 1] as usize;
    }
    // Mostly dark border -> dark background -> objects are light.
    let object_is_dark = (border_dark as f32 / (2 * (w + h)) as f32) < 0.5;

    let min_area = ((min_area_frac as f64) * n as f64).max(1.0) as usize;
    let max_area = (0.60 * n as f64) as usize;
    // `open[i]`: object pixel not yet visited.
    let mut open: Vec<bool> = dark.iter().map(|&d| d == object_is_dark).collect();
    let mut stack: Vec<usize> = Vec::new();
    let mut count = 0i64;

    for start in 0..n {
        if !open[start] {
            continue;
        }
        open[start] = false;
        stack.push(start);
        let (mut area, mut touches_border) = (0usize, false);
        while let Some(idx) = stack.pop() {
            area += 1;
            let (y, x) = (idx / w, idx % w);
            if y == 0 || x == 0 || y == h - 1 || x == w - 1 {
                touches_border = true;
            }
            let mut visit = |ni: usize| {
                if open[ni] {
                    open[ni] = false;
                    stack.push(ni);
                }
            };
            if y > 0 {
                visit(idx - w);
            }
            if y + 1 < h {
                visit(idx + w);
            }
            if x > 0 {
                visit(idx - 1);
            }
            if x + 1 < w {
                visit(idx + 1);
            }
        }
        if area >= min_area && area <= max_area && !touches_border {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blobs(h: usize, w: usize, n: usize) -> Vec<f32> {
        let mut g = vec![1.0f32; h * w];
        let cy = h / 2;
        for k in 0..n {
            let cx = (w * (k + 1)) / (n + 1);
            for dy in 0..8 {
                for dx in 0..8 {
                    g[(cy + dy - 4) * w + cx + dx - 4] = 0.0;
                }
            }
        }
        g
    }

    #[test]
    fn counts_blobs_on_light_and_dark_backgrounds() {
        for n in [1usize, 2, 3, 5] {
            assert_eq!(count_blobs(&make_blobs(64, 64, n), 64, 64, 0.0005), n as i64, "n={n}");
        }
        let inv: Vec<f32> = make_blobs(64, 64, 4).iter().map(|v| 1.0 - v).collect();
        assert_eq!(count_blobs(&inv, 64, 64, 0.0005), 4);
        assert_eq!(count_blobs(&[], 0, 0, 0.01), -1);
    }

    #[test]
    fn alpha_cumprod_is_decreasing() {
        let betas: Vec<f64> = (0..1000).map(|i| 0.0001 + (0.02 - 0.0001) * i as f64 / 999.0).collect();
        let mut out = vec![0.0f64; 1000];
        assert_eq!(unsafe { sdx_alpha_cumprod_f64(betas.as_ptr(), out.as_mut_ptr(), 1000) }, 0);
        assert!(out.windows(2).all(|p| p[1] < p[0]) && out[999] > 0.0);
    }

    #[test]
    fn jaccard_and_highlight() {
        assert_eq!(token_jaccard_basis_points("a, b", "A c"), 3333);
        assert_eq!(token_jaccard_basis_points("", "a"), 0);
        let mut rgb = vec![0u8; 8 * 8 * 3];
        rgb[..2 * 8 * 3].fill(255);
        assert!((highlight_frac(&rgb, 250.0) - 0.25).abs() < 1e-9);
    }
}
