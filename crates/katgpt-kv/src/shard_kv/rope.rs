//! RoPE (Rotary Position Embedding) undo/reapply utilities.
//!
//! RoPE applies position-dependent rotation to pairs of dimensions:
//!   For dim pair (2i, 2i+1), rotation angle = pos × inv_freq[i]
//!   where inv_freq[i] = 1.0 / (10000^(2i/d_head))
//!
//! `undo_rope` applies the INVERSE rotation (negated angles).
//! `reapply_rope` applies the FORWARD rotation.
//!
//! These are exact algebraic inverses — roundtrip error is at most float epsilon.

/// Pre-computed RoPE inverse frequencies for a given head_dim.
///
/// Caches `inv_freq[i] = 1.0 / (10000^(2i/d_head))` to avoid recomputing
/// and reallocating on every rotation call.
///
/// Also caches pre-computed sin/cos values per position to avoid redundant
/// transcendental function calls when the same position is used across layers.
pub struct RopeFreqs {
    inv_freq: Vec<f32>,
    half: usize,
    /// Last position used to fill `sincos_buf`.
    cached_pos: usize,
    /// Whether `negate` was used for the cached values.
    cached_negate: bool,
    /// Pre-computed (sin, cos) for the cached position. Reused across layers.
    sincos_buf: Vec<(f32, f32)>,
}

impl RopeFreqs {
    /// Build inverse frequencies for the given head_dim.
    pub fn new(head_dim: usize) -> Self {
        let half = head_dim / 2;
        let base: f32 = 10000.0;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exp = 2.0 * i as f32 / head_dim as f32;
                1.0 / base.powf(exp)
            })
            .collect();

        Self {
            inv_freq,
            half,
            cached_pos: usize::MAX,
            cached_negate: false,
            sincos_buf: vec![(0.0f32, 0.0f32); half],
        }
    }

    /// Apply position-dependent rotation to dim pairs in-place.
    ///
    /// For each pair (2i, 2i+1):
    ///   θ = pos × inv_freq[i]
    ///   [x0', x1'] = [[cos θ, -sin θ], [sin θ, cos θ]] @ [x0, x1]
    ///
    /// When `negate = true`, applies the inverse rotation (negated angle).
    ///
    /// Caches sin/cos for the last `(pos, negate)` pair, so calling across
    /// multiple layers with the same position reuses the cached values
    /// and avoids `O(half)` transcendental function calls.
    #[inline]
    pub fn apply(&mut self, x: &mut [f32], pos: usize, negate: bool) {
        // Recompute sin/cos only when position or sign changes.
        // This amortizes the cost across layers (same pos, different layers).
        if pos != self.cached_pos || negate != self.cached_negate {
            let sign: f32 = if negate { -1.0 } else { 1.0 };
            let pos_f = sign * pos as f32;
            for i in 0..self.half {
                let theta = pos_f * self.inv_freq[i];
                self.sincos_buf[i] = theta.sin_cos();
            }
            self.cached_pos = pos;
            self.cached_negate = negate;
        }

        // Apply rotation using cached sin/cos — no transcendental calls.
        for i in 0..self.half {
            let (sin_t, cos_t) = self.sincos_buf[i];
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = cos_t * x0 - sin_t * x1;
            x[2 * i + 1] = sin_t * x0 + cos_t * x1;
        }
    }
}

/// Apply position-dependent rotation to dim pairs in-place.
///
/// For each pair (2i, 2i+1):
///   θ = pos × inv_freq[i]
///   [x0', x1'] = [[cos θ, -sin θ], [sin θ, cos θ]] @ [x0, x1]
///
/// When `negate = true`, applies the inverse rotation (negated angle).
///
/// Prefer [`RopeFreqs::apply`] in hot paths to avoid reallocating the
/// frequency table on every call.
fn apply_rotation(x: &mut [f32], pos: usize, head_dim: usize, negate: bool) {
    let mut freqs = RopeFreqs::new(head_dim);
    freqs.apply(x, pos, negate);
}

/// Undo RoPE: apply the inverse position-dependent rotation.
///
/// For dim pair (2i, 2i+1), applies rotation by -pos × inv_freq[i].
/// This removes the position-dependent phase structure so that subsequent
/// PCA sees spatially coherent data.
pub fn undo_rope(x: &mut [f32], pos: usize, head_dim: usize) {
    apply_rotation(x, pos, head_dim, true);
}

/// Reapply RoPE: apply the forward position-dependent rotation.
///
/// After inverse PCA rotation, the reconstructed vector needs RoPE
/// reapplied to restore position-dependent structure for attention.
pub fn reapply_rope(x: &mut [f32], pos: usize, head_dim: usize) {
    apply_rotation(x, pos, head_dim, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_reapply_roundtrip() {
        let head_dim = 128;
        let pos = 42;
        let mut x: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 + 1.0).sin() * 0.5)
            .collect();
        let original = x.clone();

        undo_rope(&mut x, pos, head_dim);
        reapply_rope(&mut x, pos, head_dim);

        for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-5,
                "roundtrip failed at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_undo_changes_vector() {
        let head_dim = 64;
        let pos = 10;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();
        let original = x.clone();

        undo_rope(&mut x, pos, head_dim);

        // Should change the vector (unless degenerate)
        let diff: f32 = original
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "undo_rope should modify the vector");
    }

    #[test]
    fn test_reapply_changes_vector() {
        let head_dim = 64;
        let pos = 10;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).cos()).collect();
        let original = x.clone();

        reapply_rope(&mut x, pos, head_dim);

        let diff: f32 = original
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "reapply_rope should modify the vector");
    }

    #[test]
    fn test_identity_at_pos_zero() {
        let head_dim = 32;
        let pos = 0;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        let original = x.clone();

        // At pos=0, rotation angle = 0 → identity
        reapply_rope(&mut x, pos, head_dim);

        for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "pos=0 should be identity at [{i}]: {orig} vs {rec}"
            );
        }
    }

    #[test]
    fn test_roundtrip_various_positions() {
        let head_dim = 64;
        for pos in [0, 1, 10, 100, 511] {
            let mut x: Vec<f32> = (0..head_dim)
                .map(|i| (i as f32 + 1.0).sin() * 0.5)
                .collect();
            let original = x.clone();

            undo_rope(&mut x, pos, head_dim);
            reapply_rope(&mut x, pos, head_dim);

            for (i, (orig, rec)) in original.iter().zip(x.iter()).enumerate() {
                assert!(
                    (orig - rec).abs() < 1e-4,
                    "roundtrip failed at pos={pos}, [{i}]: {orig} vs {rec}"
                );
            }
        }
    }

    #[test]
    fn test_cached_apply_same_as_uncached() {
        // Verify that the cached RopeFreqs::apply produces identical results
        // to the uncached free functions.
        let head_dim = 64;
        let mut freqs = RopeFreqs::new(head_dim);

        for pos in [0, 1, 7, 42, 255, 511] {
            for negate in [false, true] {
                let mut x_cached: Vec<f32> = (0..head_dim)
                    .map(|i| (i as f32 + 1.0).sin() * 0.5)
                    .collect();
                let mut x_uncached = x_cached.clone();

                freqs.apply(&mut x_cached, pos, negate);
                apply_rotation(&mut x_uncached, pos, head_dim, negate);

                for (i, (a, b)) in x_cached.iter().zip(x_uncached.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "cached vs uncached mismatch at pos={pos}, negate={negate}, [{i}]: {a} vs {b}"
                    );
                }
            }
        }
    }
}
