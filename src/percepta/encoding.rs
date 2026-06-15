//! Parabolic key encoding helpers for the Percepta CHT Hull KV Cache.
//!
//! The parabolic encoding transforms a scalar key `k` into a 2D key `(kx, ky)` such that
//! the dot product `q · k` preserves the ordering of `k` when queried with a specific
//! 2D query. This enables O(log N) hard attention via convex hull tricks.
//!
//! # Mathematical basis
//!
//! Given scalar key `k` and offset (query position) `offset`:
//! - `kx = 2k - 2·offset`
//! - `ky = -k² + 2k·offset - offset² + tie_term`
//!
//! When the query `(q - offset, 1)` is dot-producted with the encoded key:
//! `q·k = -(k - offset)² + offset² + q·offset + tie_term`
//!
//! This parabolic form ensures the convex hull of encoded keys corresponds to the
//! maximum attention scores, enabling efficient retrieval.

use super::types::{HARD_K, TieBreak};

/// Encode a scalar key `k` into a 2D key for parabolic attention.
///
/// The parabolic encoding ensures that the dot product `q · k` equals
/// `-(k - offset)² + offset² + q·offset + tie_break_term` when
/// the query is properly encoded.
///
/// # Arguments
/// * `k` — scalar key value
/// * `offset` — center of the parabolic curve (typically the query position)
/// * `tie_break` — tie-breaking mode (affects the y-component)
/// * `inv_log_pos` — `1 / ln(position)` for [`TieBreak::Average`] tie-breaking
///
/// # Returns
/// 2D key `[kx, ky]`
#[inline]
pub fn encode_key(k: f64, offset: f64, tie_break: TieBreak, inv_log_pos: f64) -> [f64; 2] {
    // Hoist `d = k - offset` so both kx and ky reuse it:
    //   kx = 2*(k - offset)
    //   ky = -(k - offset)² + tie_term
    // which saves one multiply and one subtract vs the expanded form.
    let d = k - offset;
    let tie_term = match tie_break {
        TieBreak::Average => inv_log_pos,
        TieBreak::Latest => 0.0, // Latest uses sequence numbers in HullMeta
    };
    [2.0 * d, (-(d * d)) + tie_term]
}

/// Encode a scalar query `q` into a 2D query for parabolic attention.
///
/// The query is `(q - offset, 1)`. When dot-producted with the encoded key,
/// this produces the parabolic attention score.
///
/// # Arguments
/// * `q` — scalar query value
/// * `offset` — center of the parabolic curve (same offset used for key encoding)
///
/// # Returns
/// 2D query `[qx, qy]`
#[inline]
pub fn encode_query(q: f64, offset: f64) -> [f64; 2] {
    [q - offset, 1.0]
}

/// Subtract `big` from the y-component of a key.
///
/// This separates the tie-breaking term from the main key value
/// when using [`BIG`] as an offset.
///
/// # Arguments
/// * `key` — 2D key `[kx, ky]`
/// * `big` — large constant to subtract (typically [`BIG`])
///
/// # Returns
/// Key with `ky - big`
#[inline]
pub fn clear_key(key: [f64; 2], big: f64) -> [f64; 2] {
    [key[0], key[1] - big]
}

/// Scale a key by [`HARD_K`] for hard (argmax) attention behavior.
///
/// Multiplies both components by [`HARD_K`], sharpening the attention
/// distribution toward a single maximum.
///
/// # Arguments
/// * `key` — 2D key `[kx, ky]`
///
/// # Returns
/// Scaled key `[kx * HARD_K, ky * HARD_K]`
#[inline]
pub fn hard_scale(key: [f64; 2]) -> [f64; 2] {
    [key[0] * HARD_K, key[1] * HARD_K]
}

/// Scale a query by [`HARD_K`] for hard attention behavior.
///
/// Both key and query must be scaled to preserve the relative ordering
/// of dot products.
///
/// # Arguments
/// * `query` — 2D query `[qx, qy]`
///
/// # Returns
/// Scaled query `[qx * HARD_K, qy * HARD_K]`
#[inline]
pub fn hard_scale_query(query: [f64; 2]) -> [f64; 2] {
    [query[0] * HARD_K, query[1] * HARD_K]
}

#[cfg(test)]
mod tests {
    use super::super::types::BIG;
    use super::*;

    #[test]
    fn test_encode_key_basic() {
        let k = 3.0;
        let offset = 1.0;
        let key = encode_key(k, offset, TieBreak::Average, 0.5);
        // kx = 2*3 - 2*1 = 4
        // ky = -9 + 6 - 1 + 0.5 = -3.5
        assert!((key[0] - 4.0).abs() < 1e-12, "kx mismatch");
        assert!((key[1] - (-3.5)).abs() < 1e-12, "ky mismatch");
    }

    #[test]
    fn test_encode_key_latest_ignores_inv_log_pos() {
        let k = 2.0;
        let offset = 1.0;
        let key = encode_key(k, offset, TieBreak::Latest, 999.0);
        // kx = 2*2 - 2*1 = 2
        // ky = -4 + 4 - 1 + 0 = -1  (tie_term = 0 for Latest)
        assert!((key[0] - 2.0).abs() < 1e-12, "kx mismatch");
        assert!((key[1] - (-1.0)).abs() < 1e-12, "ky mismatch");
    }

    #[test]
    fn test_encode_query_basic() {
        let q = 5.0;
        let offset = 2.0;
        let query = encode_query(q, offset);
        assert!((query[0] - 3.0).abs() < 1e-12, "qx mismatch");
        assert!((query[1] - 1.0).abs() < 1e-12, "qy mismatch");
    }

    #[test]
    fn test_dot_product_parabolic_form() {
        // Verify: query · key = -(k - q)² + (q - offset)² + tie_term
        let k = 3.0;
        let q = 5.0;
        let offset = 2.0;
        let inv_log_pos = 0.1;

        let key = encode_key(k, offset, TieBreak::Average, inv_log_pos);
        let query = encode_query(q, offset);

        let dot = query[0] * key[0] + query[1] * key[1];
        let expected = -(k - q).powi(2) + (q - offset).powi(2) + inv_log_pos;
        assert!(
            (dot - expected).abs() < 1e-9,
            "dot={dot}, expected={expected}"
        );
    }

    #[test]
    fn test_clear_key() {
        let key = [3.0, 1e12 + 5.0];
        let cleared = clear_key(key, BIG);
        assert!((cleared[0] - 3.0).abs() < 1e-12, "kx should be unchanged");
        assert!(
            (cleared[1] - 5.0).abs() < 1e-6,
            "ky should be {} ~= 5.0",
            cleared[1]
        );
    }

    #[test]
    fn test_hard_scale_key() {
        let key = [2.0, -3.0];
        let scaled = hard_scale(key);
        assert!((scaled[0] - 2.0e6).abs() < 1e-6);
        assert!((scaled[1] - (-3.0e6)).abs() < 1e-6);
    }

    #[test]
    fn test_hard_scale_query() {
        let query = [1.5, 1.0];
        let scaled = hard_scale_query(query);
        assert!((scaled[0] - 1.5e6).abs() < 1e-6);
        assert!((scaled[1] - 1.0e6).abs() < 1e-6);
    }

    #[test]
    fn test_hard_scale_preserves_ordering() {
        // Two keys with different scores; hard scaling should preserve ordering
        let offset = 0.0;
        let query = encode_query(1.0, offset);
        let key_a = encode_key(1.0, offset, TieBreak::Latest, 0.0);
        let key_b = encode_key(2.0, offset, TieBreak::Latest, 0.0);

        let score_a = query[0] * key_a[0] + query[1] * key_a[1];
        let score_b = query[0] * key_b[0] + query[1] * key_b[1];

        let hq = hard_scale_query(query);
        let hk_a = hard_scale(key_a);
        let hk_b = hard_scale(key_b);

        let h_score_a = hq[0] * hk_a[0] + hq[1] * hk_a[1];
        let h_score_b = hq[0] * hk_b[0] + hq[1] * hk_b[1];

        // Same ordering
        assert_eq!(
            score_a.partial_cmp(&score_b),
            h_score_a.partial_cmp(&h_score_b),
            "hard scaling must preserve score ordering"
        );
    }

    #[test]
    fn test_encode_key_zero_offset() {
        let key = encode_key(4.0, 0.0, TieBreak::Average, 1.0);
        // kx = 2*4 - 0 = 8
        // ky = -16 + 0 - 0 + 1 = -15
        assert!((key[0] - 8.0).abs() < 1e-12);
        assert!((key[1] - (-15.0)).abs() < 1e-12);
    }

    #[test]
    fn test_encode_query_zero_offset() {
        let query = encode_query(4.0, 0.0);
        assert!((query[0] - 4.0).abs() < 1e-12);
        assert!((query[1] - 1.0).abs() < 1e-12);
    }
}
