//! InstantStyle / style-embed math — C ABI for Python ctypes.
//!
//! Kernels: weighted mean of embedding rows, content subtract `I - s*C`,
//! and in-place L2 row normalize. Complements CUDA `sdx_cuda_ml` on CPU.

use std::slice;

/// Weighted mean of `n_rows` vectors of length `dim` → `out` (length `dim`).
///
/// # Safety
/// Pointers must be valid for the stated sizes.
#[no_mangle]
pub unsafe extern "C" fn sdx_style_weighted_mean_f32(
    rows: *const f32,
    weights: *const f32,
    n_rows: usize,
    dim: usize,
    out: *mut f32,
) -> i32 {
    if rows.is_null() || weights.is_null() || out.is_null() || n_rows == 0 || dim == 0 {
        return -1;
    }
    let mat = slice::from_raw_parts(rows, n_rows * dim);
    let w = slice::from_raw_parts(weights, n_rows);
    let dst = slice::from_raw_parts_mut(out, dim);
    weighted_mean(mat, w, n_rows, dim, dst);
    0
}

/// `out[i] = image[i] - strength * content[i]` for `dim` elements.
#[no_mangle]
pub unsafe extern "C" fn sdx_style_subtract_f32(
    image: *const f32,
    content: *const f32,
    dim: usize,
    strength: f32,
    out: *mut f32,
) -> i32 {
    if image.is_null() || content.is_null() || out.is_null() || dim == 0 {
        return -1;
    }
    let a = slice::from_raw_parts(image, dim);
    let b = slice::from_raw_parts(content, dim);
    let dst = slice::from_raw_parts_mut(out, dim);
    for i in 0..dim {
        dst[i] = a[i] - strength * b[i];
    }
    0
}

/// L2-normalize `n_rows` rows of length `dim` in place. Zero rows stay zero.
#[no_mangle]
pub unsafe extern "C" fn sdx_style_l2_normalize_rows_f32(rows: *mut f32, n_rows: usize, dim: usize) -> i32 {
    if rows.is_null() || n_rows == 0 || dim == 0 {
        return -1;
    }
    let mat = slice::from_raw_parts_mut(rows, n_rows * dim);
    l2_normalize_rows(mat, n_rows, dim);
    0
}

pub fn weighted_mean(mat: &[f32], w: &[f32], n_rows: usize, dim: usize, out: &mut [f32]) {
    let mut wsum = 0.0f32;
    for &wi in w.iter().take(n_rows) {
        wsum += wi.max(0.0);
    }
    if wsum <= 1e-12 {
        out[..dim].fill(0.0);
        return;
    }
    out[..dim].fill(0.0);
    for r in 0..n_rows {
        let wi = w[r].max(0.0) / wsum;
        let base = r * dim;
        for d in 0..dim {
            out[d] += mat[base + d] * wi;
        }
    }
}

pub fn l2_normalize_rows(mat: &mut [f32], n_rows: usize, dim: usize) {
    for r in 0..n_rows {
        let base = r * dim;
        let mut s = 0.0f32;
        for d in 0..dim {
            let v = mat[base + d];
            s += v * v;
        }
        let n = s.sqrt();
        if n > 1e-12 {
            for d in 0..dim {
                mat[base + d] /= n;
            }
        }
    }
}
