//! `extract_top_k_peaks` — extracts the top-K values from a logit slice
//! in descending order.

/// Maximum K supported by stack-allocated `extract_top_k_peaks_arr`.
pub const MAX_TOP_K: usize = 16;

/// Returns the top `k` values from `logits`, sorted descending.
/// If `k` exceeds the slice length, returns all values sorted descending.
///
/// Uses `select_nth_unstable_by` for O(n) partial sort instead of full sort.
/// Allocates one copy of the input slice (needed for in-place partition).
pub fn extract_top_k_peaks(logits: &[f32], k: usize) -> Vec<f32> {
    let k = k.min(logits.len());
    if k == 0 {
        return Vec::new();
    }
    let mut values = logits.to_vec();
    // O(n) partial sort: partition so first k are the largest (unordered), then sort those k
    let _ = values.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    values.truncate(k);
    values.sort_unstable_by(|a, b| b.total_cmp(a));
    values
}

/// Zero-alloc top-K extraction into a fixed-size stack buffer.
///
/// Returns a slice of the top `k` values sorted descending.
/// `buf` must be at least `k` elements. Uses insertion-maintenance for small k
/// (k ≤ MAX_TOP_K) which is O(n * k) — faster than O(n log n) for small k.
#[inline]
pub fn extract_top_k_into<'a>(
    logits: &[f32],
    k: usize,
    buf: &'a mut [f32; MAX_TOP_K],
) -> &'a [f32] {
    let k = k.min(logits.len()).min(MAX_TOP_K);
    if k == 0 {
        return &[];
    }
    // Initialize with NEG_INFINITY
    buf[..k].fill(f32::NEG_INFINITY);

    // Maintain a sorted-descending top-k via insertion
    for &val in logits {
        if val <= buf[k - 1] {
            continue; // Skip if not in top-k
        }
        // Find insertion position (descending order)
        let pos = match buf[..k].binary_search_by(|probe| val.total_cmp(probe)) {
            Ok(idx) | Err(idx) => idx.min(k - 1),
        };
        // Shift elements down
        buf.copy_within(pos..k - 1, pos + 1);
        buf[pos] = val;
    }
    &buf[..k]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_k() {
        let logits = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let peaks = extract_top_k_peaks(&logits, 3);
        assert_eq!(peaks, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    fn k_exceeds_length() {
        let logits = vec![0.1, 0.3];
        let peaks = extract_top_k_peaks(&logits, 5);
        assert_eq!(peaks, vec![0.3, 0.1]);
    }

    #[test]
    fn empty_input() {
        let peaks = extract_top_k_peaks(&[], 3);
        assert!(peaks.is_empty());
    }

    #[test]
    fn top_k_into_matches_allocating() {
        let logits = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.6];
        let expected = extract_top_k_peaks(&logits, 4);
        let mut buf = [0.0f32; MAX_TOP_K];
        let result = extract_top_k_into(&logits, 4, &mut buf);
        assert_eq!(result, &expected[..]);
    }

    #[test]
    fn top_k_into_empty() {
        let mut buf = [0.0f32; MAX_TOP_K];
        let result = extract_top_k_into(&[], 3, &mut buf);
        assert!(result.is_empty());
    }

    #[test]
    fn top_k_into_single_element() {
        let logits = vec![0.5];
        let mut buf = [0.0f32; MAX_TOP_K];
        let result = extract_top_k_into(&logits, 3, &mut buf);
        assert_eq!(result, &[0.5]);
    }
}
