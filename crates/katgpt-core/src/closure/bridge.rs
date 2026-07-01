//! Latent bridges — raw PTG ↔ latent motif embedding ↔ scalar TaR score.
//!
//! Per AGENTS.md "Latent vs Raw Space Rules": PTG structure is raw/syncable;
//! motif embeddings are latent/local; TaR score is a diagnostic scalar that
//! may cross the sync boundary but only as audit data, never for anti-cheat
//! validation. These bridges implement the three transitions:
//!
//! 1. **raw → latent**: [`ptg_to_motif_embedding`] — per-PTG count vector
//!    projected via `sigmoid(dirs · feature_vec)` using
//!    [`crate::simd::simd_dot_f32`]. Output length = `K` directions.
//! 2. **latent → raw scalar**: [`motif_embedding_to_tar_score`] — mean of
//!    embedding components clamped to `[0,1]`. Placeholder for a learned
//!    sigmoid direction (the real version ships with riir-ai's
//!    `AnchorProfile.translate_priorities()` output).
//!
//! ## Hot-path caveat
//!
//! These functions allocate the output `Vec<f32>` internally for ergonomics.
//! Callers in the hot path should pre-allocate a scratch buffer and reuse it
//! via `Vec::clear()` + write into `&mut [T]`. (Per AGENTS.md "Allocation".)
//! The cold/warm paths (PRI/CDG/TaR aggregation, motif mining) are fine with
//! per-call allocation.

use super::{PrimitiveKind, PrimitiveTransitionGraph};
use crate::simd::simd_dot_f32;

/// Default number of motif direction vectors (K = 32). Each direction is an
/// `N`-dim vector where `N = size of the primitive enumeration space covered
/// by the directions table`. The exact `N` is configured per [`MotifDirections`].
pub const DEFAULT_MOTIF_DIRS: usize = 32;

/// Pre-computed lookup table of motif direction vectors.
///
/// `directions` is a flat `K × N` buffer: row `k` occupies
/// `directions[k*N .. (k+1)*N]`. Each row is a unit-norm (caller-enforced)
/// direction in primitive-count space.
#[derive(Clone, Debug)]
pub struct MotifDirections {
    /// `K * N` flat buffer, row-major.
    pub directions: Vec<f32>,
    /// Number of direction rows (K).
    pub k: usize,
    /// Dimensionality of each direction (N).
    pub n: usize,
}

impl MotifDirections {
    /// Build a directions table from a flat `K × N` buffer.
    ///
    /// `directions.len()` must equal `k * n` or this returns `None`.
    #[inline]
    #[must_use]
    pub fn from_flat(directions: Vec<f32>, k: usize, n: usize) -> Option<Self> {
        if directions.len() != k.checked_mul(n)? {
            return None;
        }
        Some(Self { directions, k, n })
    }

    /// Construct a zeroed directions table of the given shape. Useful for
    /// tests / deterministic fixtures.
    #[inline]
    #[must_use]
    pub fn zeros(k: usize, n: usize) -> Self {
        Self {
            directions: vec![0.0; k * n],
            k,
            n,
        }
    }

    /// Get a single direction row as a slice.
    #[inline]
    #[must_use]
    pub fn row(&self, k: usize) -> &[f32] {
        let start = k * self.n;
        &self.directions[start..start + self.n]
    }
}

/// **Raw → latent**: project a PTG into a `K`-dim motif embedding.
///
/// Builds a per-primitive-kind count vector (length `N` = `dirs.n`), then for
/// each of the `K` directions takes the dot product and applies `sigmoid`.
/// Output length is exactly `K`.
///
/// # Edge cases
///
/// - Empty PTG ⇒ all-zero count vector ⇒ all `sigmoid(0.0) = 0.5`.
/// - `dirs.n == 0` ⇒ returns `K` copies of `0.5` (the dot product of empty
///   slices is `0.0`).
///
/// # Sigmoid not softmax
///
/// Per AGENTS.md: never softmax. Each direction's projection is independent —
/// they don't compete. Sigmoid bounds each to `(0, 1)`.
#[inline]
#[must_use]
pub fn ptg_to_motif_embedding(ptg: &PrimitiveTransitionGraph, dirs: &MotifDirections) -> Vec<f32> {
    // 1. Build the count vector (length N).
    let mut feature = vec![0.0f32; dirs.n];
    for node in &ptg.nodes {
        let idx = feature_index(node.primitive, dirs.n);
        if let Some(slot) = feature.get_mut(idx) {
            *slot += 1.0;
        }
    }

    // 2. For each direction k: sigmoid(dot(dirs.row(k), &feature)).
    let mut out: Vec<f32> = Vec::with_capacity(dirs.k);
    for k in 0..dirs.k {
        let row = dirs.row(k);
        let dot = simd_dot_f32(row, &feature, dirs.n);
        out.push(sigmoid(dot));
    }
    out
}

/// **Latent → raw scalar**: project a motif embedding down to a single TaR
/// diagnostic score in `[0, 1]`.
///
/// Placeholder: returns the arithmetic mean of `emb`, clamped to `[0, 1]`.
/// The real version would project onto a learned sigmoid direction (private
/// to riir-ai's `AnchorProfile`); that direction is shipped as a model asset
/// and used here directly once available.
///
/// # Edge cases
///
/// - Empty embedding ⇒ `0.0` (no signal).
/// - Inputs already in `(0, 1)` (e.g. produced by [`ptg_to_motif_embedding`])
///   ⇒ mean is also in `(0, 1)`, clamp is a no-op.
#[inline]
#[must_use]
pub fn motif_embedding_to_tar_score(emb: &[f32]) -> f32 {
    if emb.is_empty() {
        return 0.0;
    }
    let sum: f32 = emb.iter().copied().sum();
    let mean = sum / emb.len() as f32;
    mean.clamp(0.0, 1.0)
}

/// Map a [`PrimitiveKind`] to an index in the `N`-dim feature vector.
///
/// Uses the canonical `u32` wire form modulo `N` — this is a hash into the
/// feature space, not a 1:1 enumeration. For PTGs whose primitive ids exceed
/// `N`, distinct primitives may alias; this is acceptable because the
/// downstream `K × N` projection absorbs the collision (each direction still
/// gets a distinct linear combination).
#[inline(always)]
fn feature_index(p: PrimitiveKind, n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (p.to_u32() as usize) % n
    }
}

/// Numerically stable logistic sigmoid. Per AGENTS.md: sigmoid, not softmax.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    // Branch-free form: avoids overflow on large negatives.
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::closure::{OperatorKind, PtgRecorder};

    fn build_ptg(task_family: u32, primitives: &[u32]) -> PrimitiveTransitionGraph {
        let mut rec = PtgRecorder::new(task_family);
        let mut prev: Option<u32> = None;
        for (i, &p) in primitives.iter().enumerate() {
            let n = rec.enter(PrimitiveKind::UserDefined(p), i as u32, Some([p as u8; 32]));
            if let Some(p_id) = prev {
                rec.exit(p_id, n, OperatorKind::Sequence);
            }
            prev = Some(n);
        }
        rec.finish()
    }

    #[test]
    fn embedding_output_length_is_k() {
        let dirs = MotifDirections::zeros(8, 5);
        let ptg = build_ptg(0, &[0, 1, 2]);
        let emb = ptg_to_motif_embedding(&ptg, &dirs);
        assert_eq!(emb.len(), 8);
    }

    #[test]
    fn embedding_outputs_in_unit_interval() {
        // Random-ish directions.
        let mut dirs_vec: Vec<f32> = Vec::with_capacity(4 * 6);
        for i in 0..(4 * 6) {
            dirs_vec.push((i as f32 * 0.37) - 1.5); // mix of signs and magnitudes
        }
        let dirs = MotifDirections::from_flat(dirs_vec, 4, 6).expect("shape");
        let ptg = build_ptg(0, &[0, 1, 2, 3]);
        let emb = ptg_to_motif_embedding(&ptg, &dirs);
        for v in &emb {
            assert!(*v >= 0.0 && *v <= 1.0, "emb value out of [0,1]: {v}");
        }
    }

    #[test]
    fn empty_ptg_yields_sigmoid_zero() {
        // All-zero feature ⇒ dot = 0 for every direction ⇒ sigmoid(0) = 0.5.
        let dirs = MotifDirections::zeros(4, 5);
        let ptg = PrimitiveTransitionGraph::empty(0);
        let emb = ptg_to_motif_embedding(&ptg, &dirs);
        for v in &emb {
            assert!((v - 0.5).abs() < 1e-6, "expected sigmoid(0)=0.5, got {v}");
        }
    }

    #[test]
    fn tar_scalar_in_unit_interval_and_clamps() {
        // Out-of-range input.
        let big: Vec<f32> = vec![5.0, 7.0, 9.0];
        let s = motif_embedding_to_tar_score(&big);
        assert_eq!(s, 1.0, "mean of >1 must clamp to 1");
        let neg: Vec<f32> = vec![-3.0, -1.0];
        let s2 = motif_embedding_to_tar_score(&neg);
        assert_eq!(s2, 0.0, "mean of <0 must clamp to 0");
        // In-range.
        let normal: Vec<f32> = vec![0.4, 0.6];
        let s3 = motif_embedding_to_tar_score(&normal);
        assert!((s3 - 0.5).abs() < 1e-6, "mean 0.5, got {s3}");
        // Empty.
        assert_eq!(motif_embedding_to_tar_score(&[]), 0.0);
    }

    #[test]
    fn from_flat_rejects_mismatched_shape() {
        assert!(MotifDirections::from_flat(vec![1.0, 2.0, 3.0], 2, 2).is_none());
        assert!(MotifDirections::from_flat(vec![1.0, 2.0, 3.0, 4.0], 2, 2).is_some());
    }

    #[test]
    fn row_indexing_is_correct() {
        let dirs = MotifDirections::from_flat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        assert_eq!(dirs.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(dirs.row(1), &[4.0, 5.0, 6.0]);
    }

    /// Bridge round-trip: PTG → embedding → scalar; output is bounded in [0,1].
    #[test]
    fn bridge_pipeline_stays_bounded() {
        let dirs = MotifDirections::from_flat(
            (0..(DEFAULT_MOTIF_DIRS * 8)).map(|i| (i as f32) * 0.1 - 4.0).collect(),
            DEFAULT_MOTIF_DIRS,
            8,
        )
        .unwrap();
        let ptg = build_ptg(0, &[0, 1, 2, 3, 4]);
        let emb = ptg_to_motif_embedding(&ptg, &dirs);
        let scalar = motif_embedding_to_tar_score(&emb);
        assert!((0.0..=1.0).contains(&scalar), "scalar={scalar}");
    }
}
