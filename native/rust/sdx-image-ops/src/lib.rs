//! Deterministic image ops — C ABI cdylib for Python ctypes.
//!
//! `sdx_count_blobs_f32` counts salient foreground objects in a single-channel
//! image via Otsu thresholding + connected-components labeling, inferring the
//! background from the image border (robust to bright- or dark-background
//! images). It mirrors the NumPy reference in
//! `scripts/tools/eval/scorers.py::_count_blobs_numpy` but runs GIL-free in a
//! tight loop with an explicit stack (no recursion, no allocation per pixel).
//!
//! Used by the count-adherence scorer in the eval harness: "exactly N objects"
//! is a prompt constraint no leading text-to-image model guarantees, so a fast
//! deterministic counter lets the self-critique loop resample failing outputs.
//!
//! # Safety
//! All exported functions take raw pointers with explicit lengths. Callers must
//! ensure pointers are valid for the stated element counts.

use std::slice;

/// Otsu threshold over a 256-bin histogram of values in `[0, 1]`.
/// Returns a threshold in `[0, 1]`, placed between bins so a class sitting
/// exactly on the optimal bin lands on the intended side of the comparison.
fn otsu_threshold(gray: &[f32]) -> f32 {
    let mut hist = [0u64; 256];
    for &v in gray {
        let mut b = (v * 255.0).round() as i32;
        if b < 0 {
            b = 0;
        } else if b > 255 {
            b = 255;
        }
        hist[b as usize] += 1;
    }
    let total = gray.len() as f64;
    let mut sum_all = 0.0f64;
    for (t, &h) in hist.iter().enumerate() {
        sum_all += t as f64 * h as f64;
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut best_t = 0usize;
    let mut best_var = -1.0f64;
    for t in 0..256 {
        w_b += hist[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * hist[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t;
        }
    }
    (best_t as f32 + 0.5) / 255.0
}

/// Count foreground blobs in a row-major `h*w` grayscale image (`f32` in [0,1]).
///
/// * `min_area_frac` — components smaller than this fraction of the image are
///   ignored (noise). Components larger than 60% of the image, or touching the
///   border, are treated as background and not counted.
///
/// Returns the blob count (>= 0), or -1 on invalid arguments.
///
/// # Safety
/// `gray` must point to at least `h * w` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn sdx_count_blobs_f32(
    gray: *const f32,
    h: usize,
    w: usize,
    min_area_frac: f32,
) -> i64 {
    if gray.is_null() || h == 0 || w == 0 {
        return -1;
    }
    let n = h * w;
    let g = slice::from_raw_parts(gray, n);
    count_blobs(g, h, w, min_area_frac)
}

/// Pure-Rust core (also usable as a library via `crate-type = rlib`).
pub fn count_blobs(g: &[f32], h: usize, w: usize, min_area_frac: f32) -> i64 {
    if h == 0 || w == 0 {
        return -1;
    }
    let n = h * w;
    if g.len() < n {
        return -1;
    }
    let thr = otsu_threshold(g);

    // `dark[i]` = pixel below threshold.
    let dark = |i: usize| g[i] < thr;

    // Background = class dominating the border ring.
    let mut border_dark = 0usize;
    let mut border_total = 0usize;
    for x in 0..w {
        border_dark += dark(x) as usize; // top row
        border_dark += dark((h - 1) * w + x) as usize; // bottom row
        border_total += 2;
    }
    for y in 0..h {
        border_dark += dark(y * w) as usize; // left col
        border_dark += dark(y * w + (w - 1)) as usize; // right col
        border_total += 2;
    }
    let border_dark_frac = if border_total > 0 {
        border_dark as f32 / border_total as f32
    } else {
        0.5
    };
    // If the border is mostly dark, background is dark -> objects are light.
    let object_is_dark = border_dark_frac < 0.5;
    let is_object = |i: usize| dark(i) == object_is_dark;

    let min_area = ((min_area_frac as f64) * n as f64).max(1.0) as usize;
    let max_area = (0.60 * n as f64) as usize;

    let mut labels = vec![0i32; n];
    let mut cur = 0i32;
    let mut count: i64 = 0;
    let mut stack: Vec<usize> = Vec::new();

    for start in 0..n {
        if is_object(start) && labels[start] == 0 {
            cur += 1;
            let mut area = 0usize;
            let mut touches_border = false;
            stack.push(start);
            labels[start] = cur;
            while let Some(idx) = stack.pop() {
                area += 1;
                let y = idx / w;
                let x = idx % w;
                if y == 0 || x == 0 || y == h - 1 || x == w - 1 {
                    touches_border = true;
                }
                // 4-connectivity neighbours.
                if y > 0 {
                    let ni = idx - w;
                    if is_object(ni) && labels[ni] == 0 {
                        labels[ni] = cur;
                        stack.push(ni);
                    }
                }
                if y + 1 < h {
                    let ni = idx + w;
                    if is_object(ni) && labels[ni] == 0 {
                        labels[ni] = cur;
                        stack.push(ni);
                    }
                }
                if x > 0 {
                    let ni = idx - 1;
                    if is_object(ni) && labels[ni] == 0 {
                        labels[ni] = cur;
                        stack.push(ni);
                    }
                }
                if x + 1 < w {
                    let ni = idx + 1;
                    if is_object(ni) && labels[ni] == 0 {
                        labels[ni] = cur;
                        stack.push(ni);
                    }
                }
            }
            if area >= min_area && area <= max_area && !touches_border {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render `n` non-touching square blobs on a white background and count them.
    fn make_blobs(h: usize, w: usize, n: usize) -> Vec<f32> {
        let mut g = vec![1.0f32; h * w]; // white bg
        let cy = h / 2;
        for k in 0..n {
            let cx = (w * (k + 1)) / (n + 1);
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = cy + dy - 4;
                    let x = cx + dx - 4;
                    g[y * w + x] = 0.0; // dark blob
                }
            }
        }
        g
    }

    #[test]
    fn counts_light_bg_dark_blobs() {
        for n in [1usize, 2, 3, 5] {
            let g = make_blobs(64, 64, n);
            assert_eq!(count_blobs(&g, 64, 64, 0.0005), n as i64, "n={n}");
        }
    }

    #[test]
    fn counts_dark_bg_light_blobs() {
        // Invert: dark bg, light blobs.
        let mut g = make_blobs(64, 64, 4);
        for v in g.iter_mut() {
            *v = 1.0 - *v;
        }
        assert_eq!(count_blobs(&g, 64, 64, 0.0005), 4);
    }

    #[test]
    fn rejects_bad_args() {
        assert_eq!(count_blobs(&[], 0, 0, 0.01), -1);
    }
}
