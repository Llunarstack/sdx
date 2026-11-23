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

// ---------------------------------------------------------------------------------------------
// Image kernels on float32 HWC buffers (values nominally in [0, 1]), used by `utils.ideas`.
// Return 0 on success, -2 on bad arguments. Parallel kernels use at most four threads.
// ---------------------------------------------------------------------------------------------

/// Call `f(y, row)` for every output row of `dst` (`row_len` floats per row), in up to four row bands.
fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(dst: &mut [f32], h: usize, row_len: usize, f: F) {
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 4).min(h.max(1));
    let band = h.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        for (k, chunk) in dst.chunks_mut(band * row_len).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (r, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(k * band + r, row);
                }
            });
        }
    });
}

/// In-place Floyd-Steinberg (serpentine when `serpentine != 0`) quantisation of each channel to `levels` steps;
/// `amount` in [0, 1] scales the diffused error (0 = plain rounding).
#[no_mangle]
pub unsafe extern "C" fn sdx_error_diffuse_f32(
    img: *mut f32,
    h: usize,
    w: usize,
    c: usize,
    levels: u32,
    amount: f32,
    serpentine: i32,
) -> i32 {
    if img.is_null() || h == 0 || w == 0 || c == 0 || levels < 2 {
        return -2;
    }
    let buf = slice::from_raw_parts_mut(img, h * w * c);
    let q = (levels - 1) as f32;
    let amt = amount.clamp(0.0, 1.0);
    for y in 0..h {
        let rev = serpentine != 0 && y % 2 == 1;
        let (fwd, back): (isize, isize) = if rev { (-1, 1) } else { (1, -1) };
        for k in 0..w {
            let x = if rev { w - 1 - k } else { k };
            for ch in 0..c {
                let i = (y * w + x) * c + ch;
                let old = buf[i];
                let new = (old * q).round().clamp(0.0, q) / q;
                buf[i] = new;
                let err = (old - new) * amt;
                for (dx, dy, wgt) in [(fwd, 0usize, 7.0f32), (back, 1, 3.0), (0, 1, 5.0), (fwd, 1, 1.0)] {
                    let nx = x as isize + dx;
                    if nx >= 0 && (nx as usize) < w && y + dy < h {
                        buf[((y + dy) * w + nx as usize) * c + ch] += err * wgt / 16.0;
                    }
                }
            }
        }
    }
    0
}

/// `n x n` blue-noise threshold map in (0, 1) by void-and-cluster (Ulichney 1993) on a torus, Gaussian `sigma`.
#[no_mangle]
pub unsafe extern "C" fn sdx_blue_noise_f32(out: *mut f32, n: usize, sigma: f32, seed: u64) -> i32 {
    if out.is_null() || !(4..=256).contains(&n) || !(sigma > 0.0) {
        return -2;
    }
    let ranks = void_and_cluster(n, sigma, seed);
    let size = (n * n) as f32;
    for (o, &r) in slice::from_raw_parts_mut(out, n * n).iter_mut().zip(&ranks) {
        *o = (r as f32 + 0.5) / size;
    }
    0
}

fn void_and_cluster(n: usize, sigma: f32, seed: u64) -> Vec<u32> {
    let size = n * n;
    let mut kern = vec![0f32; size];
    for dy in 0..n {
        for dx in 0..n {
            let (a, b) = (dy.min(n - dy) as f32, dx.min(n - dx) as f32);
            kern[dy * n + dx] = (-(a * a + b * b) / (2.0 * sigma * sigma)).exp();
        }
    }
    let splat = |energy: &mut [f32], p: usize, sign: f32| {
        let (py, px) = (p / n, p % n);
        for y in 0..n {
            let row = &kern[((y + n - py) % n) * n..][..n];
            for x in 0..n {
                energy[y * n + x] += sign * row[(x + n - px) % n];
            }
        }
    };
    let pick = |pattern: &[bool], energy: &[f32], ones: bool| -> usize {
        let mut best = usize::MAX;
        for p in 0..size {
            if pattern[p] == ones
                && (best == usize::MAX || (ones && energy[p] > energy[best]) || (!ones && energy[p] < energy[best]))
            {
                best = p;
            }
        }
        best
    };
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut rand = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut pattern = vec![false; size];
    let mut energy = vec![0f32; size];
    let mut ones = 0;
    while ones < (size / 10).max(1) {
        let p = (rand() % size as u64) as usize;
        if !pattern[p] {
            pattern[p] = true;
            splat(&mut energy, p, 1.0);
            ones += 1;
        }
    }
    for _ in 0..size {
        let cluster = pick(&pattern, &energy, true);
        pattern[cluster] = false;
        splat(&mut energy, cluster, -1.0);
        let void = pick(&pattern, &energy, false);
        pattern[void] = true;
        splat(&mut energy, void, 1.0);
        if void == cluster {
            break;
        }
    }
    let mut rank = vec![0u32; size];
    let (mut proto, mut e) = (pattern.clone(), energy.clone());
    for r in (0..ones).rev() {
        let p = pick(&proto, &e, true);
        proto[p] = false;
        splat(&mut e, p, -1.0);
        rank[p] = r as u32;
    }
    let (mut proto, mut e) = (pattern, energy);
    for r in ones..size {
        let p = pick(&proto, &e, false);
        proto[p] = true;
        splat(&mut e, p, 1.0);
        rank[p] = r as u32;
    }
    rank
}

fn clamp_at(v: isize, hi: usize) -> usize {
    v.clamp(0, hi as isize - 1) as usize
}

/// Edge-preserving bilateral filter: spatial Gaussian `sigma_s` (px) times range Gaussian `sigma_r` on the
/// colour distance, over a `(2 * radius + 1)^2` window with clamped borders.
#[no_mangle]
pub unsafe extern "C" fn sdx_bilateral_f32(
    src: *const f32,
    dst: *mut f32,
    h: usize,
    w: usize,
    c: usize,
    radius: usize,
    sigma_s: f32,
    sigma_r: f32,
) -> i32 {
    if src.is_null() || dst.is_null() || h == 0 || w == 0 || !(1..=4).contains(&c) || !(sigma_s > 0.0 && sigma_r > 0.0) {
        return -2;
    }
    let src = slice::from_raw_parts(src, h * w * c);
    let dst = slice::from_raw_parts_mut(dst, h * w * c);
    let r = radius as isize;
    let spatial: Vec<f32> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (-((dx * dx + dy * dy) as f32) / (2.0 * sigma_s * sigma_s)).exp()))
        .collect();
    let inv = 1.0 / (2.0 * sigma_r * sigma_r);
    par_rows(dst, h, w * c, |y, row| {
        for x in 0..w {
            let center = &src[(y * w + x) * c..][..c];
            let (mut acc, mut norm, mut k) = ([0f32; 4], 0f32, 0usize);
            for dy in -r..=r {
                let yy = clamp_at(y as isize + dy, h);
                for dx in -r..=r {
                    let p = &src[(yy * w + clamp_at(x as isize + dx, w)) * c..][..c];
                    let d2: f32 = p.iter().zip(center).map(|(a, b)| (a - b) * (a - b)).sum();
                    let wgt = spatial[k] * (-d2 * inv).exp();
                    k += 1;
                    for ch in 0..c {
                        acc[ch] += wgt * p[ch];
                    }
                    norm += wgt;
                }
            }
            for ch in 0..c {
                row[x * c + ch] = acc[ch] / norm;
            }
        }
    });
    0
}

/// 3x3 median per channel with clamped borders.
#[no_mangle]
pub unsafe extern "C" fn sdx_median3_f32(src: *const f32, dst: *mut f32, h: usize, w: usize, c: usize) -> i32 {
    if src.is_null() || dst.is_null() || h == 0 || w == 0 || c == 0 {
        return -2;
    }
    let src = slice::from_raw_parts(src, h * w * c);
    let dst = slice::from_raw_parts_mut(dst, h * w * c);
    par_rows(dst, h, w * c, |y, row| {
        let mut win = [0f32; 9];
        for x in 0..w {
            for ch in 0..c {
                let mut k = 0;
                for dy in -1..=1isize {
                    let yy = clamp_at(y as isize + dy, h);
                    for dx in -1..=1isize {
                        win[k] = src[(yy * w + clamp_at(x as isize + dx, w)) * c + ch];
                        k += 1;
                    }
                }
                win.select_nth_unstable_by(4, |a, b| a.total_cmp(b));
                row[x * c + ch] = win[4];
            }
        }
    });
    0
}

/// Kuwahara filter: each pixel takes the mean colour of whichever of its four `(radius + 1)^2` quadrants has the
/// lowest luma variance (summed-area tables, so O(1) per pixel).
#[no_mangle]
pub unsafe extern "C" fn sdx_kuwahara_f32(src: *const f32, dst: *mut f32, h: usize, w: usize, c: usize, radius: usize) -> i32 {
    if src.is_null() || dst.is_null() || h == 0 || w == 0 || c == 0 || radius == 0 {
        return -2;
    }
    let src = slice::from_raw_parts(src, h * w * c);
    let dst = slice::from_raw_parts_mut(dst, h * w * c);
    let (iw, planes) = (w + 1, c + 2);
    let mut sat = vec![0f64; (h + 1) * iw * planes];
    for y in 0..h {
        for x in 0..w {
            let p = &src[(y * w + x) * c..][..c];
            let l = if c >= 3 { 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] } else { p[0] } as f64;
            for k in 0..planes {
                let v = if k < c { p[k] as f64 } else if k == c { l } else { l * l };
                let at = |yy: usize, xx: usize| ((yy * iw) + xx) * planes + k;
                sat[at(y + 1, x + 1)] = v + sat[at(y, x + 1)] + sat[at(y + 1, x)] - sat[at(y, x)];
            }
        }
    }
    let rect = |k: usize, y0: usize, x0: usize, y1: usize, x1: usize| {
        let at = |yy: usize, xx: usize| sat[((yy * iw) + xx) * planes + k];
        at(y1 + 1, x1 + 1) - at(y0, x1 + 1) - at(y1 + 1, x0) + at(y0, x0)
    };
    par_rows(dst, h, w * c, |y, row| {
        for x in 0..w {
            let (ylo, yhi) = (y.saturating_sub(radius), (y + radius).min(h - 1));
            let (xlo, xhi) = (x.saturating_sub(radius), (x + radius).min(w - 1));
            let mut best = (f64::INFINITY, (y, x, y, x));
            for q in [(ylo, xlo, y, x), (ylo, x, y, xhi), (y, xlo, yhi, x), (y, x, yhi, xhi)] {
                let n = ((q.2 - q.0 + 1) * (q.3 - q.1 + 1)) as f64;
                let mean = rect(c, q.0, q.1, q.2, q.3) / n;
                let var = rect(c + 1, q.0, q.1, q.2, q.3) / n - mean * mean;
                if var < best.0 {
                    best = (var, q);
                }
            }
            let q = best.1;
            let n = ((q.2 - q.0 + 1) * (q.3 - q.1 + 1)) as f64;
            for ch in 0..c {
                row[x * c + ch] = (rect(ch, q.0, q.1, q.2, q.3) / n) as f32;
            }
        }
    });
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_diffusion_preserves_mean_and_quantises() {
        let (h, w) = (32, 32);
        let mut img = vec![0.3f32; h * w];
        assert_eq!(unsafe { sdx_error_diffuse_f32(img.as_mut_ptr(), h, w, 1, 2, 1.0, 1) }, 0);
        assert!(img.iter().all(|&v| v == 0.0 || v == 1.0));
        let mean = img.iter().sum::<f32>() / (h * w) as f32;
        assert!((mean - 0.3).abs() < 0.02, "mean {mean}");
    }

    #[test]
    fn blue_noise_ranks_are_a_permutation() {
        let mut out = vec![0f32; 16 * 16];
        assert_eq!(unsafe { sdx_blue_noise_f32(out.as_mut_ptr(), 16, 1.5, 3) }, 0);
        let mut ranks: Vec<u32> = out.iter().map(|v| (v * 256.0) as u32).collect();
        ranks.sort_unstable();
        assert_eq!(ranks, (0..256).collect::<Vec<u32>>());
    }

    #[test]
    fn median_removes_a_hot_pixel_and_kuwahara_keeps_flat() {
        let (h, w) = (5, 5);
        let mut src = vec![0.5f32; h * w];
        src[12] = 1.0;
        let mut dst = vec![0f32; h * w];
        assert_eq!(unsafe { sdx_median3_f32(src.as_ptr(), dst.as_mut_ptr(), h, w, 1) }, 0);
        assert_eq!(dst[12], 0.5);
        let flat = vec![0.25f32; h * w * 3];
        let mut out = vec![0f32; h * w * 3];
        assert_eq!(unsafe { sdx_kuwahara_f32(flat.as_ptr(), out.as_mut_ptr(), h, w, 3, 2) }, 0);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

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
