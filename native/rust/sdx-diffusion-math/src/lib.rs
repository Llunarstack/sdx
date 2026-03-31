//! VP-diffusion math helpers — C ABI cdylib for Python ctypes.
//!
//! Matches the NumPy implementations in `diffusion/snr_utils.py` and
//! `diffusion/schedules.py` but runs in a tight Rust loop with no GIL,
//! no NumPy overhead, and no Python object allocation.
//!
//! # Safety
//! All exported functions take raw pointers with explicit lengths.
//! Callers must ensure pointers are valid and non-overlapping where required.

use std::slice;

// ---------------------------------------------------------------------------
// alpha_cumprod
// ---------------------------------------------------------------------------

/// Compute `alpha_cumprod[t] = prod_{s<=t} (1 - beta[s])` in-place into `out`.
///
/// # Safety
/// `betas` and `out` must each point to at least `n` valid `f64` values.
/// `betas` and `out` must not overlap.
#[no_mangle]
pub unsafe extern "C" fn sdx_alpha_cumprod_f64(
    betas: *const f64,
    out: *mut f64,
    n: usize,
) -> i32 {
    if betas.is_null() || out.is_null() || n == 0 {
        return -2;
    }
    let b = slice::from_raw_parts(betas, n);
    let o = slice::from_raw_parts_mut(out, n);
    let mut acc: f64 = 1.0;
    for (i, &beta) in b.iter().enumerate() {
        acc *= 1.0 - beta;
        o[i] = acc;
    }
    0
}

// ---------------------------------------------------------------------------
// SNR from alpha_cumprod
// ---------------------------------------------------------------------------

/// Compute `snr[t] = alpha_cumprod[t] / (1 - alpha_cumprod[t] + 1e-8)`.
///
/// # Safety
/// `alpha_cumprod` and `out` must each point to at least `n` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn sdx_snr_from_alpha_cumprod_f64(
    alpha_cumprod: *const f64,
    out: *mut f64,
    n: usize,
) -> i32 {
    if alpha_cumprod.is_null() || out.is_null() || n == 0 {
        return -2;
    }
    let ac = slice::from_raw_parts(alpha_cumprod, n);
    let o = slice::from_raw_parts_mut(out, n);
    for (i, &a) in ac.iter().enumerate() {
        o[i] = a / (1.0 - a + 1e-8);
    }
    0
}

// ---------------------------------------------------------------------------
// Linear beta schedule
// ---------------------------------------------------------------------------

/// Fill `out[0..n]` with a linear beta schedule from `beta_start` to `beta_end`.
///
/// # Safety
/// `out` must point to at least `n` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn sdx_linear_beta_schedule_f64(
    out: *mut f64,
    n: usize,
    beta_start: f64,
    beta_end: f64,
) -> i32 {
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

// ---------------------------------------------------------------------------
// Squared-cosine v2 beta schedule
// ---------------------------------------------------------------------------

/// Fill `out[0..n]` with the squared-cosine v2 beta schedule.
/// `alpha_bar(t) = cos^2((t/T + 0.008) / 1.008 * pi/2)`.
///
/// # Safety
/// `out` must point to at least `n` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn sdx_squaredcos_beta_schedule_v2_f64(
    out: *mut f64,
    n: usize,
    max_beta: f64,
) -> i32 {
    if out.is_null() || n == 0 {
        return -2;
    }
    let o = slice::from_raw_parts_mut(out, n);
    let nf = n as f64;
    let pi_half = std::f64::consts::PI * 0.5;
    for i in 0..n {
        let t1 = i as f64 / nf;
        let t2 = (i + 1) as f64 / nf;
        let ab1 = ((t1 + 0.008) / 1.008 * pi_half).cos().powi(2);
        let ab2 = ((t2 + 0.008) / 1.008 * pi_half).cos().powi(2);
        let beta = (1.0 - ab2 / ab1.max(1e-12)).min(max_beta);
        o[i] = beta.clamp(1e-4, 0.999);
    }
    0
}

// ---------------------------------------------------------------------------
// Cosine beta schedule
// ---------------------------------------------------------------------------

/// Fill `out[0..n]` with the cosine beta schedule (Nichol & Dhariwal style).
///
/// # Safety
/// `out` must point to at least `n` valid `f64` values.
#[no_mangle]
pub unsafe extern "C" fn sdx_cosine_beta_schedule_f64(
    out: *mut f64,
    n: usize,
) -> i32 {
    if out.is_null() || n == 0 {
        return -2;
    }
    let o = slice::from_raw_parts_mut(out, n);
    let nf = n as f64;
    let pi_half = std::f64::consts::PI * 0.5;

    // alpha_bar[i] = cos^2(((i/n) + 0.01) / 1.01 * pi/2)
    let ab = |i: usize| -> f64 {
        let t = i as f64 / nf;
        ((t + 0.01) / 1.01 * pi_half).cos().powi(2)
    };
    let ab0 = ab(0);
    for i in 0..n {
        let a1 = ab(i) / ab0;
        let a2 = ab(i + 1) / ab0;
        let beta = (1.0 - a2 / a1.max(1e-12)).clamp(1e-4, 0.999);
        o[i] = beta;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_cumprod_monotone() {
        let betas: Vec<f64> = (0..1000)
            .map(|i| 0.0001 + (0.02 - 0.0001) * i as f64 / 999.0)
            .collect();
        let mut out = vec![0.0f64; 1000];
        unsafe { sdx_alpha_cumprod_f64(betas.as_ptr(), out.as_mut_ptr(), 1000) };
        // alpha_cumprod must be strictly decreasing
        for i in 1..1000 {
            assert!(out[i] < out[i - 1], "not monotone at {i}");
        }
        assert!(out[999] > 0.0);
        assert!(out[0] < 1.0);
    }

    #[test]
    fn test_snr_positive() {
        let ac: Vec<f64> = (0..100).map(|i| 1.0 - i as f64 / 100.0).collect();
        let mut snr = vec![0.0f64; 100];
        unsafe { sdx_snr_from_alpha_cumprod_f64(ac.as_ptr(), snr.as_mut_ptr(), 100) };
        for &s in &snr {
            assert!(s >= 0.0);
        }
    }
}
