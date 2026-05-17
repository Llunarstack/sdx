//! Caption CSV merge/dedupe and positive/negative word conflict filter — C ABI for Python.

use std::collections::HashSet;
use std::slice;
use std::str;

fn split_comma_segments(text: &str) -> Vec<&str> {
    text.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}

fn positive_word_set(positive: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for part in positive.split(',') {
        for word in part.split_whitespace() {
            let w = word.trim();
            if !w.is_empty() {
                set.insert(w.to_ascii_lowercase());
            }
        }
    }
    set
}

fn token_set_lower(text: &str) -> HashSet<String> {
    let mut set = HashSet::new();
    for part in text.split(',') {
        for word in part.split_whitespace() {
            let w = word.trim();
            if !w.is_empty() {
                set.insert(w.to_ascii_lowercase());
            }
        }
    }
    set
}

/// Jaccard similarity in basis points (0–10000).
fn token_jaccard_basis_points(a: &str, b: &str) -> i32 {
    let sa = token_set_lower(a);
    let sb = token_set_lower(b);
    if sa.is_empty() && sb.is_empty() {
        return 0;
    }
    if sa.is_empty() || sb.is_empty() {
        return 0;
    }
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    if union == 0 {
        return 0;
    }
    ((inter as f64 / union as f64) * 10000.0).round() as i32
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn merge_caption_dedupe(parts: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for p in parts {
        for s in split_comma_segments(p.as_ref()) {
            let key = s.to_ascii_lowercase();
            if seen.insert(key) {
                out.push(s.to_string());
            }
        }
    }
    out.join(", ")
}

fn filter_negative(positive: &str, negative: &str) -> String {
    let pos_set = positive_word_set(positive);
    if pos_set.is_empty() {
        return negative.to_string();
    }
    let mut kept: Vec<String> = Vec::new();
    for part in negative.split(',') {
        let words: Vec<&str> = part
            .split_whitespace()
            .filter(|w| !w.is_empty() && !pos_set.contains(&w.to_ascii_lowercase()))
            .collect();
        if !words.is_empty() {
            kept.push(words.join(" "));
        }
    }
    let result = kept.join(", ").trim().to_string();
    if result.is_empty() {
        " ".to_string()
    } else {
        result
    }
}

fn write_utf8_out(s: &str, out: *mut u8, out_cap: usize) -> i64 {
    let bytes = s.as_bytes();
    if out.is_null() {
        return bytes.len() as i64;
    }
    if out_cap < bytes.len() {
        return -1;
    }
    unsafe {
        slice::from_raw_parts_mut(out, bytes.len()).copy_from_slice(bytes);
    }
    bytes.len() as i64
}

/// Required output capacity (or bytes written if `out` is non-null and large enough).
///
/// # Safety
/// `pos`/`neg` must be valid UTF-8 for `pos_len`/`neg_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn sdx_filter_negative_utf8(
    pos: *const u8,
    pos_len: usize,
    neg: *const u8,
    neg_len: usize,
    out: *mut u8,
    out_cap: usize,
) -> i64 {
    if pos.is_null() || neg.is_null() {
        return -2;
    }
    let pos_s = match str::from_utf8(slice::from_raw_parts(pos, pos_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let neg_s = match str::from_utf8(slice::from_raw_parts(neg, neg_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let filtered = filter_negative(pos_s, neg_s);
    write_utf8_out(&filtered, out, out_cap)
}

/// Merge comma-separated caption fragments with case-insensitive dedupe (first casing kept).
#[no_mangle]
pub unsafe extern "C" fn sdx_merge_caption_utf8(
    a: *const u8,
    a_len: usize,
    b: *const u8,
    b_len: usize,
    out: *mut u8,
    out_cap: usize,
) -> i64 {
    if a.is_null() || b.is_null() {
        return -2;
    }
    let a_s = match str::from_utf8(slice::from_raw_parts(a, a_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let b_s = match str::from_utf8(slice::from_raw_parts(b, b_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let merged = merge_caption_dedupe([a_s, b_s]);
    write_utf8_out(&merged, out, out_cap)
}

/// Token-set Jaccard similarity in basis points (0–10000). Returns -3 on bad UTF-8.
#[no_mangle]
pub unsafe extern "C" fn sdx_token_jaccard_utf8(
    a: *const u8,
    a_len: usize,
    b: *const u8,
    b_len: usize,
) -> i32 {
    if a.is_null() || b.is_null() {
        return -2;
    }
    let a_s = match str::from_utf8(slice::from_raw_parts(a, a_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    let b_s = match str::from_utf8(slice::from_raw_parts(b, b_len)) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    token_jaccard_basis_points(a_s, b_s)
}

/// FNV-1a 64-bit fingerprint of UTF-8 bytes (style genome / caption id).
#[no_mangle]
pub unsafe extern "C" fn sdx_fnv1a64_utf8(data: *const u8, data_len: usize) -> u64 {
    if data.is_null() || data_len == 0 {
        return 0;
    }
    fnv1a64(slice::from_raw_parts(data, data_len))
}

/// Merge up to 8 comma-separated UTF-8 fragments with dedupe (for style axis bundles).
#[no_mangle]
pub unsafe extern "C" fn sdx_merge_style_axes_utf8(
    parts: *const *const u8,
    lengths: *const usize,
    count: usize,
    out: *mut u8,
    out_cap: usize,
) -> i64 {
    if parts.is_null() || lengths.is_null() || count == 0 || count > 8 {
        return -2;
    }
    let mut strings: Vec<&str> = Vec::with_capacity(count);
    for i in 0..count {
        let ptr = unsafe { *parts.add(i) };
        let len = unsafe { *lengths.add(i) };
        if ptr.is_null() {
            continue;
        }
        match str::from_utf8(unsafe { slice::from_raw_parts(ptr, len) }) {
            Ok(s) if !s.trim().is_empty() => strings.push(s),
            Ok(_) => {}
            Err(_) => return -3,
        }
    }
    let merged = merge_caption_dedupe(strings);
    write_utf8_out(&merged, out, out_cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_drops_pos_words() {
        let out = filter_negative("red dress, blue sky", "red, blurry, dress");
        assert!(!out.to_lowercase().contains("red"));
        assert!(!out.to_lowercase().contains("dress"));
        assert!(out.contains("blurry"));
    }

    #[test]
    fn merge_dedupes() {
        let m = merge_caption_dedupe(["a, b", "B, c"].iter().map(|s| *s));
        assert_eq!(m, "a, b, c");
    }

    #[test]
    fn jaccard_identical() {
        let j = token_jaccard_basis_points("red dress, blue", "red dress, blue");
        assert_eq!(j, 10000);
    }

    #[test]
    fn fnv_stable() {
        let h1 = fnv1a64(b"glitch cathedral");
        let h2 = fnv1a64(b"glitch cathedral");
        assert_eq!(h1, h2);
        assert_ne!(h1, fnv1a64(b"other"));
    }
}
