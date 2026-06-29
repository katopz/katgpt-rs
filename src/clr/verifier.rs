//! `SigmoidProjectionVerifier` — dot-product + sigmoid [`ClaimVerifier`] (Plan 284 T1.5).
//!
//! This is the only verifier required for Phase 1. It computes
//! `sigmoid(dot(claim.embedding, direction_vec[idx]))` using the shared SIMD
//! dot kernel (`katgpt_core::simd::simd_dot_f32`) and a scalar sigmoid (`f32::exp`).
//!
//! # Risk #3: direction-vector saturation
//!
//! Poorly-scaled direction vectors cause sigmoid saturation. If
//! `|dot(emb, dir)| >> 0`, the sigmoid output collapses to `0` or `1` and the
//! gradient information used by freeze/thaw + MGPO updates is destroyed.
//!
//! Mitigations (caller-side):
//!   1. Normalize direction vectors to unit L2 norm before passing them in.
//!   2. Bound claim embeddings (e.g. layernorm before claim extraction).
//!   3. Monitor the empirical verdict distribution; if >95% of verdicts land
//!      in `[0, 0.05] ∪ [0.95, 1]`, the directions are saturated.
//!
//! NO softmax is used anywhere in this module — only sigmoid (per project
//! convention and the user's `AGENTS.md` rule).

use crate::clr::traits::{ClaimVerifier, DirectionVectorSource};
use crate::clr::types::{Claim, Verdict};

/// Sigmoid-projection verifier: `sigmoid(dot(emb, dir[idx]))`.
///
/// Borrows a [`DirectionVectorSource`] for the lifetime of the verifier.
/// Generic over the claim payload `T` because the verdict computation only
/// touches `claim.embedding`.
pub struct SigmoidProjectionVerifier<'a> {
    /// Borrowed direction-vector pool.
    pub directions: &'a dyn DirectionVectorSource,
    /// Expected dimension `k` of each direction vector (and each claim embedding).
    pub direction_dim: usize,
}

impl<'a> SigmoidProjectionVerifier<'a> {
    /// Construct a new verifier backed by `directions` with claim/dir dim `k`.
    #[inline]
    pub fn new(directions: &'a dyn DirectionVectorSource, direction_dim: usize) -> Self {
        assert!(
            direction_dim > 0,
            "SigmoidProjectionVerifier: direction_dim must be > 0"
        );
        Self {
            directions,
            direction_dim,
        }
    }
}

impl<T> ClaimVerifier<T> for SigmoidProjectionVerifier<'_> {
    /// Compute the sigmoid verdict for `claim` projected onto direction `idx`.
    #[inline(always)]
    fn verify(&self, claim: &Claim<T>, direction_idx: usize) -> Verdict {
        let d = self.directions.direction(direction_idx);
        debug_assert_eq!(
            d.len(),
            self.direction_dim,
            "SigmoidProjectionVerifier: direction dim mismatch (got {}, expected {})",
            d.len(),
            self.direction_dim
        );
        debug_assert_eq!(
            claim.embedding.len(),
            self.direction_dim,
            "SigmoidProjectionVerifier: claim embedding dim mismatch (got {}, expected {})",
            claim.embedding.len(),
            self.direction_dim
        );
        // SIMD dot on the full k-dim vectors.
        let dot = katgpt_core::simd::simd_dot_f32(&claim.embedding, d, self.direction_dim);
        // Scalar sigmoid — single value, not a vector path; f32::exp is fine.
        sigmoid(dot)
    }
}

/// Numerically stable logistic sigmoid: `1 / (1 + exp(-x))`.
///
/// Scalar path — for batched sigmoid over a slice, use
/// `katgpt_core::simd::simd_exp_inplace` on a negated slice. We do NOT use softmax
/// anywhere (per project convention).
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clr::traits::DirectionVectorSource;
    use blake3::Hasher;

    /// Minimal direction source backed by a single flat `Vec<f32>` of
    /// `m * dim` floats, row-major.
    struct FlatDirections {
        dim: usize,
        vectors: Vec<f32>,
    }

    impl FlatDirections {
        fn from_rows(rows: &[&[f32]]) -> Self {
            let dim = rows[0].len();
            let mut vectors = Vec::with_capacity(rows.len() * dim);
            for r in rows {
                assert_eq!(r.len(), dim);
                vectors.extend_from_slice(r);
            }
            Self { dim, vectors }
        }
    }

    impl DirectionVectorSource for FlatDirections {
        fn direction(&self, idx: usize) -> &[f32] {
            &self.vectors[idx * self.dim..(idx + 1) * self.dim]
        }
        fn blake3(&self) -> [u8; 32] {
            let mut h = Hasher::new();
            h.update(bytemuck::cast_slice(&self.vectors));
            let mut out = [0u8; 32];
            out.copy_from_slice(h.finalize().as_bytes());
            out
        }
        fn version(&self) -> u64 {
            1
        }
    }

    #[test]
    fn verify_returns_sigmoid_of_dot_product() {
        // Direction 0 = [1, 0, 0, 0]. Claim embedding = [2.0, 0, 0, 0].
        // dot = 2.0. sigmoid(2.0) = 1/(1+e^-2) ≈ 0.8808.
        let dirs = FlatDirections::from_rows(&[&[1.0, 0.0, 0.0, 0.0]]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 4);
        let claim = Claim::<()> {
            embedding: vec![2.0, 0.0, 0.0, 0.0],
            payload: (),
        };
        let v = verifier.verify(&claim, 0);
        let expected = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((v - expected).abs() < 1e-6, "got {}, expected {}", v, expected);
    }

    #[test]
    fn verify_is_sigmoid_not_softmax() {
        // Softmax over multiple directions would normalize so the verdicts sum
        // to 1. Sigmoid does NOT — each direction is independent. Construct two
        // directions and a claim that has positive dot product with both; the
        // two verdicts should each be > 0.5 and their sum should be > 1 (which
        // softmax would forbid).
        let dirs = FlatDirections::from_rows(&[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
        ]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 4);
        let claim = Claim::<()> {
            embedding: vec![2.0, 3.0, 0.0, 0.0],
            payload: (),
        };
        let v0 = verifier.verify(&claim, 0); // sigmoid(2.0)
        let v1 = verifier.verify(&claim, 1); // sigmoid(3.0)
        assert!(v0 > 0.5 && v1 > 0.5);
        assert!(v0 + v1 > 1.0, "sigmoid sum {} > 1 violates softmax; this confirms sigmoid", v0 + v1);
    }

    #[test]
    fn verify_zero_dot_returns_half() {
        // dot = 0 → sigmoid(0) = 0.5 exactly.
        let dirs = FlatDirections::from_rows(&[&[1.0, 0.0]]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 2);
        let claim = Claim::<()> {
            embedding: vec![0.0, 5.0], // orthogonal to dir 0
            payload: (),
        };
        let v = verifier.verify(&claim, 0);
        assert!((v - 0.5).abs() < 1e-6, "orthogonal claim should give 0.5, got {}", v);
    }

    #[test]
    fn verify_negative_dot_below_half() {
        let dirs = FlatDirections::from_rows(&[&[1.0, 0.0]]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 2);
        let claim = Claim::<()> {
            embedding: vec![-3.0, 0.0],
            payload: (),
        };
        let v = verifier.verify(&claim, 0);
        assert!(v < 0.5, "negative dot should give < 0.5, got {}", v);
    }

    #[test]
    #[should_panic(expected = "direction_dim must be > 0")]
    fn new_panics_on_zero_dim() {
        let dirs = FlatDirections::from_rows(&[&[1.0]]);
        let _ = SigmoidProjectionVerifier::new(&dirs, 0);
    }
}
