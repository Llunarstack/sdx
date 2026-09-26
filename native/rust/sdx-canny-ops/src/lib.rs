//! Canny edge detection — C ABI cdylib for Creative Co-Pilot control maps.
//!
//! Pipeline: grayscale → 5×5 Gaussian → Sobel → non-max suppression →
//! double-threshold hysteresis. Soft-edge mode blurs the Canny result.
//!
//! # Safety
//! Exported functions take raw pointers; callers must pass valid lengths.

use std::slice;

fn clamp_u8(v: f32) -> u8 {
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v.round() as u8
    }
}

fn to_gray(rgb: &[u8], h: usize, w: usize, channels: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; h * w];
    if channels == 1 {
        for i in 0..(h * w) {
            g[i] = rgb[i] as f32;
        }
        return g;
    }
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * channels;
            let r = rgb[i] as f32;
            let gch = rgb[i + 1] as f32;
            let bch = rgb[i + 2.min(channels - 1)] as f32;
            g[y * w + x] = 0.299 * r + 0.587 * gch + 0.114 * bch;
        }
    }
    g
}

fn gaussian5(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    // Separable approx of 5×5 Gaussian (σ≈1.0): [1,4,6,4,1]/16
    let k = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let mut tmp = vec![0.0f32; h * w];
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (di, &kv) in k.iter().enumerate() {
                let xx = (x as isize + di as isize - 2).clamp(0, w as isize - 1) as usize;
                acc += src[y * w + xx] * kv;
            }
            tmp[y * w + x] = acc / 16.0;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (di, &kv) in k.iter().enumerate() {
                let yy = (y as isize + di as isize - 2).clamp(0, h as isize - 1) as usize;
                acc += tmp[yy * w + x] * kv;
            }
            out[y * w + x] = acc / 16.0;
        }
    }
    out
}

fn sobel(src: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mag = vec![0.0f32; h * w];
    let mut ang = vec![0.0f32; h * w];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let i = |yy: usize, xx: usize| src[yy * w + xx];
            let gx = -i(y - 1, x - 1) + i(y - 1, x + 1) - 2.0 * i(y, x - 1)
                + 2.0 * i(y, x + 1)
                - i(y + 1, x - 1)
                + i(y + 1, x + 1);
            let gy = -i(y - 1, x - 1) - 2.0 * i(y - 1, x) - i(y - 1, x + 1)
                + i(y + 1, x - 1)
                + 2.0 * i(y + 1, x)
                + i(y + 1, x + 1);
            let m = (gx * gx + gy * gy).sqrt();
            mag[y * w + x] = m;
            ang[y * w + x] = gy.atan2(gx);
        }
    }
    (mag, ang)
}

fn non_max(mag: &[f32], ang: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; h * w];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let a = ang[idx];
            // Quantize to 4 directions.
            let deg = a.to_degrees().rem_euclid(180.0);
            let (dx1, dy1, dx2, dy2) = if (deg >= 0.0 && deg < 22.5) || (deg >= 157.5) {
                (1isize, 0isize, -1isize, 0isize)
            } else if deg < 67.5 {
                (1, -1, -1, 1)
            } else if deg < 112.5 {
                (0, -1, 0, 1)
            } else {
                (-1, -1, 1, 1)
            };
            let n1 = mag[((y as isize + dy1) as usize) * w + (x as isize + dx1) as usize];
            let n2 = mag[((y as isize + dy2) as usize) * w + (x as isize + dx2) as usize];
            let m = mag[idx];
            if m >= n1 && m >= n2 {
                out[idx] = m;
            }
        }
    }
    out
}

fn hysteresis(nms: &[f32], h: usize, w: usize, low: f32, high: f32) -> Vec<u8> {
    let mut strong = vec![0u8; h * w];
    let mut weak = vec![false; h * w];
    for i in 0..(h * w) {
        if nms[i] >= high {
            strong[i] = 255;
        } else if nms[i] >= low {
            weak[i] = true;
        }
    }
    // Propagate strong edges into weak neighbors (stack DFS).
    let mut stack: Vec<usize> = (0..h * w).filter(|&i| strong[i] == 255).collect();
    while let Some(i) = stack.pop() {
        let y = i / w;
        let x = i % w;
        for dy in -1isize..=1 {
            for dx in -1isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let yy = y as isize + dy;
                let xx = x as isize + dx;
                if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if weak[j] && strong[j] == 0 {
                    strong[j] = 255;
                    stack.push(j);
                }
            }
        }
    }
    strong
}

fn box_blur_u8(src: &[u8], h: usize, w: usize, radius: usize) -> Vec<u8> {
    if radius == 0 {
        return src.to_vec();
    }
    let mut out = vec![0u8; h * w];
    let r = radius as isize;
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0u32;
            let mut n = 0u32;
            for dy in -r..=r {
                for dx in -r..=r {
                    let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    acc += src[yy * w + xx] as u32;
                    n += 1;
                }
            }
            out[y * w + x] = (acc / n.max(1)) as u8;
        }
    }
    out
}

/// Run Canny on RGB/gray `u8` buffer → grayscale edges (`out` length `h*w`).
///
/// * `mode`: 0 = hard Canny, 1 = soft-edge (blurred Canny).
/// * `low`/`high`: hysteresis thresholds on gradient magnitude (typical 40/100).
///
/// Returns 0 on success, -1 on bad args.
///
/// # Safety
/// `rgb` must be `h*w*channels` bytes; `out` must be `h*w` bytes.
#[no_mangle]
pub unsafe extern "C" fn sdx_canny_u8(
    rgb: *const u8,
    h: usize,
    w: usize,
    channels: usize,
    low: f32,
    high: f32,
    mode: i32,
    out: *mut u8,
) -> i32 {
    if rgb.is_null() || out.is_null() || h == 0 || w == 0 || !(1..=4).contains(&channels) {
        return -1;
    }
    let n = h * w * channels;
    let src = slice::from_raw_parts(rgb, n);
    let dst = slice::from_raw_parts_mut(out, h * w);
    let edges = canny_u8(src, h, w, channels, low, high, mode);
    dst.copy_from_slice(&edges);
    0
}

pub fn canny_u8(rgb: &[u8], h: usize, w: usize, channels: usize, low: f32, high: f32, mode: i32) -> Vec<u8> {
    let gray = to_gray(rgb, h, w, channels);
    let blur = gaussian5(&gray, h, w);
    let (mag, ang) = sobel(&blur, h, w);
    let nms = non_max(&mag, &ang, h, w);
    let lo = low.max(0.0);
    let hi = high.max(lo + 1.0);
    let mut edges = hysteresis(&nms, h, w, lo, hi);
    if mode == 1 {
        edges = box_blur_u8(&edges, h, w, 1);
        // Re-stretch soft edges toward white for ControlNet visibility.
        for v in edges.iter_mut() {
            *v = clamp_u8((*v as f32) * 1.35);
        }
    }
    edges
}
