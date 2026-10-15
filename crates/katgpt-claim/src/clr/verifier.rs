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
//!
//! # Calibration (riir-ai Issue 964 C1, Bench 807)
//!
//! These verdicts are BOUNDED in `(0,1)` by construction, but boundedness is
//! not calibration: whether a 0.8 verdict means "~80% of such claims come
//! true" depends on the direction vectors and embedding scale, and drifts as
//! either side moves. The `clr_calibration` feature ships
//! [`crate::clr::calibration::CalibratedVerifier`], which wraps this (or any)
//! verifier with the Platt-style `SigmoidGateCalibrator` (katgpt-core Issue
//! 810): observe `(sigmoid_output, outcome)` pairs where ground truth is
//! known, refit off-hot-path, and apply in front of the reliability gate —
//! cold start bit-identical, fitted transform strictly monotone. GOAT (Bench
//! 807): planted-drift fixture ECE 0.0924 → 0.0164 with planted-transform
//! recovery; the already-calibrated G2 fixture undisturbed (ΔECE +0.0002);
//! vote winners stable; observe+apply zero-alloc.

use crate::clr::traits::{ClaimVerifier, DirectionVectorSource};
use crate::clr::types::Verdict;

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
    /// Compute the sigmoid verdict for `embedding` projected onto direction `idx`.
    ///
    /// This is the core computation. [`Self::verify`] delegates here via
    /// `claim.embedding`.
    #[inline(always)]
    fn verify_embedding(&self, embedding: &[f32], direction_idx: usize) -> Verdict {
        let d = self.directions.direction(direction_idx);
        debug_assert_eq!(
            d.len(),
            self.direction_dim,
            "SigmoidProjectionVerifier: direction dim mismatch (got {}, expected {})",
            d.len(),
            self.direction_dim
        );
        debug_assert_eq!(
            embedding.len(),
            self.direction_dim,
            "SigmoidProjectionVerifier: embedding dim mismatch (got {}, expected {})",
            embedding.len(),
            self.direction_dim
        );
        // Issue 940 CL-2: ordered fold, not the platform lane kernel — the
        // sigmoid of this dot is the per-claim verdict, the atomic input of
        // the reliability aggregation (CL-1) and the vote winner; a
        // platform-lane reduction would make the verdict node-dependent.
        let dot = katgpt_core::simd::dot_f32_ordered(embedding, d);
        // Scalar sigmoid — single value, not a vector path; f32::exp is fine.
        sigmoid(dot)
    }
}

/// Numerically stable logistic sigmoid: `1 / (1 + exp(-x))`.
///
/// Delegates to [`katgpt_core::simd::fast_sigmoid`] (Cephes 6th-order polynomial
/// for `exp`, ~1 ULP accurate, ~1.7× faster than libm `exp` on aarch64). The
/// prior branching form avoided `(-x).exp()` overflow for very negative `x`;
/// Cephes handles this by construction via its `n < -126` early-exit, so the
/// branch is unnecessary. Result is in `(0, 1)` for all finite inputs.
///
/// Scalar path — for batched sigmoid over a slice, use
/// `katgpt_core::simd::simd_exp_inplace` on a negated slice. We do NOT use softmax
/// anywhere (per project convention).
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clr::traits::DirectionVectorSource;
    use crate::clr::types::Claim;
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
        assert!((v - expected).abs() < 1e-6, "got {v}, expected {expected}");
    }

    #[test]
    fn verify_is_sigmoid_not_softmax() {
        // Softmax over multiple directions would normalize so the verdicts sum
        // to 1. Sigmoid does NOT — each direction is independent. Construct two
        // directions and a claim that has positive dot product with both; the
        // two verdicts should each be > 0.5 and their sum should be > 1 (which
        // softmax would forbid).
        let dirs = FlatDirections::from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 4);
        let claim = Claim::<()> {
            embedding: vec![2.0, 3.0, 0.0, 0.0],
            payload: (),
        };
        let v0 = verifier.verify(&claim, 0); // sigmoid(2.0)
        let v1 = verifier.verify(&claim, 1); // sigmoid(3.0)
        assert!(v0 > 0.5 && v1 > 0.5);
        assert!(
            v0 + v1 > 1.0,
            "sigmoid sum {} > 1 violates softmax; this confirms sigmoid",
            v0 + v1
        );
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
        assert!(
            (v - 0.5).abs() < 1e-6,
            "orthogonal claim should give 0.5, got {v}"
        );
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
        assert!(v < 0.5, "negative dot should give < 0.5, got {v}");
    }

    #[test]
    #[should_panic(expected = "direction_dim must be > 0")]
    fn new_panics_on_zero_dim() {
        let dirs = FlatDirections::from_rows(&[&[1.0]]);
        let _ = SigmoidProjectionVerifier::new(&dirs, 0);
    }

    /// Issue 940 CL-2 pin: the verdict dot folds in ascending index order,
    /// not the platform lane kernel. The embedding is the certified len-16
    /// cancellation craft against a ones direction — ordered dot 3.0, any
    /// reassociating backend 6.0 — so a reversion to `simd_dot_f32` flips
    /// the verdict bits and this reds.
    #[test]
    fn verdict_dot_folds_in_index_order_not_platform_order() {
        use katgpt_core::simd::{dot_f32_ordered, fast_sigmoid, simd_dot_f32};

        let mut emb = [0.0f32; 16];
        emb[0] = 1e8;
        emb[1] = 1.0;
        emb[2] = 1.0;
        emb[3] = 1.0;
        emb[4] = -1e8;
        emb[5] = 1.0;
        emb[6] = 1.0;
        emb[7] = 1.0;
        let ones = [1.0f32; 16];

        // The craft is load-bearing: the two folds disagree on this input.
        assert_ne!(
            dot_f32_ordered(&emb, &ones),
            simd_dot_f32(&emb, &ones, 16),
            "craft no longer diverges: the pin stopped being load-bearing"
        );

        let dirs = FlatDirections::from_rows(&[&ones]);
        let verifier = SigmoidProjectionVerifier::new(&dirs, 16);
        // T is not inferable from the args — name it via the trait.
        use crate::clr::traits::ClaimVerifier;
        let v = ClaimVerifier::<()>::verify_embedding(&verifier, &emb, 0);
        assert_eq!(
            v.to_bits(),
            fast_sigmoid(dot_f32_ordered(&emb, &ones)).to_bits(),
            "verdict is not sigmoid of the ordered fold (platform-kernel regression?)"
        );
        // The platform value is a DIFFERENT verdict — the flip is observable.
        assert_ne!(v.to_bits(), fast_sigmoid(6.0f32).to_bits());
    }
}
