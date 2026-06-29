//! Density-budget interpolator `K(ca)`.
//!
//! Maps a context-awareness scalar `ca ∈ [0, 1]` to an integer top-K budget on
//! `[k_sparse, k_dense]`. One scalar in, one scalar out — the bridge between
//! "how much does the receiver already know" (ca) and "how many KV groups to
//! surface" (K).

// Re-export the struct so `mod.rs` can do `pub use budget::DensityBudget`.
// The struct itself lives in `types.rs`; the impl block is split across files
// (constructor + Default in types.rs, the interpolator here). Rust permits
// splitting `impl` blocks across modules within the same crate.
pub use super::types::DensityBudget;

impl DensityBudget {
    /// `K(ca) = round(k_sparse + ca · (k_dense − k_sparse))`, clamped to
    /// `[1, d_total]`.
    ///
    /// Monotone non-decreasing in `ca ∈ [0, 1]`. Saturates at the sparse floor
    /// for `ca ≤ 0` and the dense ceiling for `ca ≥ 1`. NaN is treated as `ca = 0`
    /// (sparse floor) — `f32::clamp` returns NaN for NaN input, which would
    /// propagate, so we guard explicitly.
    #[inline]
    pub fn k_for(&self, ca: f32) -> usize {
        // Explicit NaN guard: f32::clamp(NaN, 0.0, 1.0) returns NaN, not the
        // lower bound. Map NaN → 0.0 (sparse floor) for a deterministic,
        // replayable budget.
        let ca_clamped = if ca.is_nan() {
            0.0_f32
        } else {
            ca.clamp(0.0, 1.0)
        };
        let span = (self.k_dense as f32) - (self.k_sparse as f32);
        let k = (self.k_sparse as f32) + ca_clamped * span;
        let k = k.round() as i64;
        // i64 round-trip lets us clamp into [1, d_total] without overflow at the
        // usize boundary on extreme inputs (already guarded by clamp above, but
        // belt-and-braces).
        
        k.max(1).min(self.d_total as i64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ca_monotone_and_bounded() {
        let b = DensityBudget::for_dim(64);
        // 1000-point sweep: k_for must be non-decreasing in ca.
        let steps = 1000;
        let mut prev = b.k_for(0.0);
        for i in 0..=steps {
            let ca = i as f32 / steps as f32;
            let k = b.k_for(ca);
            assert!(
                k >= prev,
                "non-monotone at ca={ca}: k={k} < prev={prev}"
            );
            assert!(k >= 1, "below floor of 1 at ca={ca}");
            assert!(k <= b.d_total, "above d_total at ca={ca}");
            prev = k;
        }
    }

    #[test]
    fn test_endpoints_anchor() {
        let b = DensityBudget::for_dim(64);
        assert_eq!(b.k_for(0.0), b.k_sparse);
        assert_eq!(b.k_for(1.0), b.k_dense);
    }

    #[test]
    fn test_midpoint_within_range() {
        let b = DensityBudget::for_dim(64);
        let mid = b.k_for(0.5);
        assert!(mid >= b.k_sparse && mid <= b.k_dense);
    }

    #[test]
    fn test_clamping_handles_out_of_range_and_nan() {
        let b = DensityBudget::for_dim(64);
        assert_eq!(b.k_for(-0.5), b.k_sparse);
        assert_eq!(b.k_for(2.0), b.k_dense);
        // NaN is explicitly mapped to ca=0.0 (sparse floor) — `f32::clamp`
        // returns NaN for NaN input (NOT the lower bound), so `k_for` guards
        // it explicitly. See the impl docstring.
        assert_eq!(b.k_for(f32::NAN), b.k_sparse);
    }

    #[test]
    fn test_degenerate_dim_one() {
        let b = DensityBudget::for_dim(1);
        // d=1 → k_sparse=k_dense=1, interpolator pinned at 1 for all ca.
        for ca in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(b.k_for(ca), 1, "d=1 should pin K=1 at ca={ca}");
        }
    }
}
