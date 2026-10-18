//! Subspace Steering Field — k-dim manifold steering primitive (Plan 412).
//!
//! The k-dim generalization of [`LatentSteeringVector`](crate::latent_steering::LatentSteeringVector)
//! (Plan 309). The existing primitive is strictly 1D (`direction: Vec<f32>` +
//! scalar `α`, math `s' = s + α·v`). This module generalizes it to a k-dim
//! orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, with math
//! `s' = s + Σ_j α_j · u_j`. At `K=1` it is bit-identical to Plan 309; at
//! `K≥2` it enables **manifold walking** — sweeping `alphas` over a grid to
//! generate concept variations (the Goodfire "pretzel manifold" pattern,
//! adapted to our latent-state substrate).
//!
//! The block basis comes from **pre-discovered** sources (Plan 301 Jacobian
//! SVD, SpectralQuant offline eigenbasis, or hand-constructed orthogonal sets)
//! — no training at inference. The primitive is the *consumer* of discovered
//! blocks, not the featurizer trainer (that's riir-train).
//!
//! # Const generics
//!
//! [`SubspaceSteeringField<const D, const K>`] is parameterized by:
//! - `D` — the latent dimension (e.g. 8 for HLA, 64 for shards)
//! - `K` — the block size (number of orthonormal directions)
//!
//! All storage is fixed-size arrays (`[[f32; D]; K]`, `[f32; K]`, `[u8; 32]`),
//! so the struct is stack-only and zero-alloc by construction.
//!
//! # K=1 parity contract (the load-bearing gate)
//!
//! `SubspaceSteeringField<D, 1>` with a Plan 309 direction + alpha must produce
//! bit-identical output to [`crate::latent_steering::apply_latent_steering`]. This is verified by the
//! `k1_parity_with_plan_309` unit test and is the foundation of the GOAT gate
//! (Plan 412 Phase 4 T4.5 expands it to 100 random pairs).
//!
//! # References
//!
//! - Plan: `.plans/412_subspace_steering_field_primitive.md`
//! - Research: `.research/393_Block_Sparse_Featurizer_Subspace_Concept_Primitive.md`
//! - Source paper: [arXiv:2606.25234](https://arxiv.org/abs/2606.25234) —
//!   Goodfire, Block-Sparse Featurizers
//! - 1D sibling: Plan 309 (`latent_steering.rs`, DEFAULT-ON)
//!
//! Subspace Steering Field — k-dim manifold steering (Plan 412, Research 393,
//! arxiv 2606.25234 Goodfire BSF). The k-dim generalization of Plan 309: an
//! orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, math
//! `s' = s + Σ_j α_j · u_j`. At K=1 bit-identical to Plan 309; at K≥2 enables
//! manifold walking (sweep alphas over a grid → concept variations). Pure
//! modelless consumer of pre-discovered blocks (Plan 301 Jacobian SVD,
//! SpectralQuant eigenbasis, or hand-constructed sets). Phase 3 finding:
//! Newton-Schulz DIVERGES on non-square K<D matrices → Gram-Schmidt is the
//! orthonormalize constructor. DEFAULT-ON (Plan 412 Phase 5, 2026-07-08):
//! G1+G3+G4+G5 ALL PASS — G1 K=1 parity 0 mismatches/800 comparisons, G3 0
//! allocs/1000 calls × K={1,2,4}, G4 sizes 68/104/176 bytes exact, G5
//! commitment + walk_manifold deterministic.

use blake3::Hasher;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by [`SubspaceSteeringField::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SubspaceSteeringError {
    /// A block row is not unit-norm within `orthonormal_tol`, OR two rows are
    /// not mutually orthogonal within `orthonormal_tol` (off-diagonal of
    /// `block · blockᵀ` exceeds the tolerance).
    NotOrthonormal,
    /// A per-axis strength `α_j` is outside `[0.0, 1.0]`.
    AlphaOutOfRange,
}

// ── Field ───────────────────────────────────────────────────────────────────

/// A k-dim orthonormal block + per-axis strengths, BLAKE3-committed.
///
/// Generalization of [`LatentSteeringVector`](crate::latent_steering::LatentSteeringVector)
/// (Plan 309) from 1D to k-dim. At `K=1` this is bit-identical to Plan 309
/// (single direction + scalar α). At `K≥2` it enables manifold walking — sweep
/// `alphas` over a grid to generate concept variations within the subspace.
///
/// The block basis `{u_1..u_k}` is PRE-DISCOVERED (Plan 301 Jacobian SVD,
/// SpectralQuant offline eigenbasis, or hand-constructed orthogonal set). No
/// training at inference. The primitive is the *consumer* of discovered blocks,
/// not the featurizer trainer.
///
/// # Construction
///
/// Use [`new`](Self::new) to validate orthonormality + alpha range. Use
/// [`from_directions_orthonormalize`](Self::from_directions_orthonormalize)
/// to apply Newton-Schulz orthogonalization (Plan 152) to a set of
/// nearly-orthogonal directions first.
///
/// # Allocation
///
/// Zero heap allocations by construction — `block`, `alphas`, and `commitment`
/// are all fixed-size arrays. `size_of::<SubspaceSteeringField<D, K>>()` ==
/// `K*D*4 + K*4 + 32` bytes.
#[derive(Debug, Clone)]
pub struct SubspaceSteeringField<const D: usize, const K: usize> {
    /// Orthonormal block basis, row-major. Each `block[j]` is unit-norm and
    /// orthogonal to `block[i]` for `i != j`. Constructed via
    /// [`new`](Self::new) (validated) or
    /// [`from_directions_orthonormalize`](Self::from_directions_orthonormalize)
    /// (Newton-Schulz).
    pub block: [[f32; D]; K],
    /// Per-axis strengths `α_j ∈ [0, 1]`, sigmoid-bounded at construction.
    pub alphas: [f32; K],
    /// `BLAKE3(block_le || alphas_le)` — content-addressed commitment.
    pub commitment: [u8; 32],
}

impl<const D: usize, const K: usize> SubspaceSteeringField<D, K> {
    /// Construct a steering field, validating orthonormality and alpha range.
    ///
    /// Returns [`SubspaceSteeringError::NotOrthonormal`] if any `block[j]` is
    /// not unit-norm within `orthonormal_tol`, or if any pair `block[i]`,
    /// `block[j]` (`i != j`) has `|dot| > orthonormal_tol` (not orthogonal).
    /// Returns [`SubspaceSteeringError::AlphaOutOfRange`] if any `alphas[j]`
    /// is outside `[0, 1]`.
    pub fn new(
        block: [[f32; D]; K],
        alphas: [f32; K],
        orthonormal_tol: f32,
    ) -> Result<Self, SubspaceSteeringError> {
        for &a in &alphas {
            if !(0.0..=1.0).contains(&a) {
                return Err(SubspaceSteeringError::AlphaOutOfRange);
            }
        }
        // Each row must be unit-norm.
        for row in &block {
            let norm = row_norm(row);
            if (norm - 1.0).abs() > orthonormal_tol {
                return Err(SubspaceSteeringError::NotOrthonormal);
            }
        }
        // Each pair of distinct rows must be orthogonal (dot ~ 0).
        for i in 0..K {
            for j in (i + 1)..K {
                let dot = dot_product(&block[i], &block[j]);
                if dot.abs() > orthonormal_tol {
                    return Err(SubspaceSteeringError::NotOrthonormal);
                }
            }
        }
        let commitment = compute_block_commitment(&block, &alphas);
        Ok(Self {
            block,
            alphas,
            commitment,
        })
    }

    /// Construct a steering field from a set of (nearly-orthogonal) directions,
    /// applying Gram-Schmidt orthogonalization first.
    ///
    /// Convenience constructor for callers who have a set of nearly-orthogonal
    /// directions (e.g. from Plan 301 Jacobian SVD or a hand-constructed set
    /// with small numerical drift) and want them cleaned up into a proper
    /// orthonormal block before constructing the field.
    ///
    /// # Algorithm choice: Gram-Schmidt, NOT Newton-Schulz
    ///
    /// The plan (T3.1) specified Newton-Schulz (Plan 152), but empirical
    /// testing (2026-07-08) found that Newton-Schulz **diverges** on non-square
    /// K<D matrices: the Muon-tuned coefficients `(a,b,c) = (3.4445, -4.7750,
    /// 2.0315)` are designed for square weight-gradient matrices, and on a
    /// K=2 D=8 input the iteration produced dot products that GREW with more
    /// iterations (5 iters: dot=0.10; 10 iters: dot=-0.30; 15 iters: dot=0.37).
    /// Gram-Schmidt is exact, stable, and the standard algorithm for K<D
    /// orthonormalization — it's the right tool here.
    ///
    /// After orthogonalization, validates that the result is orthonormal within
    /// `orthonormal_tol` and that all `alphas` are in `[0, 1]`. Returns
    /// [`SubspaceSteeringError`] on validation failure.
    ///
    /// # Allocation
    ///
    /// Zero heap allocations — the Gram-Schmidt pass works in-place on the
    /// stack-allocated `[[f32; D]; K]` block. The resulting field is stack-only.
    pub fn from_directions_orthonormalize(
        directions: &[[f32; D]; K],
        alphas: [f32; K],
        orthonormal_tol: f32,
    ) -> Result<Self, SubspaceSteeringError> {
        // Gram-Schmidt: for each row, subtract its projection onto every
        // already-orthonormalized row, then normalize to unit norm.
        let mut block = *directions;
        for i in 0..K {
            // Subtract projections onto rows 0..i (already orthonormalized).
            // Copy each prev row to a local to avoid borrowing block[i] while
            // also reading block[..i] (the borrow checker can't prove the
            // disjointness of block[i] from block[0..i]).
            for j in 0..i {
                let prev = block[j]; // copy out
                let proj = dot_product(&block[i], &prev);
                for d in 0..D {
                    block[i][d] -= proj * prev[d];
                }
            }
            // Normalize to unit norm.
            let norm = row_norm(&block[i]);
            if norm > f32::EPSILON {
                for x in block[i].iter_mut() {
                    *x /= norm;
                }
            }
        }
        Self::new(block, alphas, orthonormal_tol)
    }

    /// Construct without validation. Caller guarantees orthonormality +
    /// alpha range. Used when the block comes from a trusted frozen artifact.
    pub fn new_unchecked(block: [[f32; D]; K], alphas: [f32; K]) -> Self {
        let commitment = compute_block_commitment(&block, &alphas);
        Self {
            block,
            alphas,
            commitment,
        }
    }

    /// Re-check orthonormality (within `tol`) AND that the stored commitment
    /// matches the current contents. Returns `false` if either check fails.
    pub fn verify(&self, tol: f32) -> bool {
        for row in &self.block {
            let norm = row_norm(row);
            if (norm - 1.0).abs() > tol {
                return false;
            }
        }
        for i in 0..K {
            for j in (i + 1)..K {
                let dot = dot_product(&self.block[i], &self.block[j]);
                if dot.abs() > tol {
                    return false;
                }
            }
        }
        compute_block_commitment(&self.block, &self.alphas) == self.commitment
    }

    /// Apply the field to a single latent state slice. Zero-alloc.
    ///
    /// Computes `state[j] += Σ_k alphas[k] * block[k][j]` over `K·D` elements.
    /// At `K=1` this reduces to Plan 309's
    /// [`apply_latent_steering`](crate::latent_steering::apply_latent_steering).
    ///
    /// Written as an explicit nested loop (outer over K, inner over D) to keep
    /// the shape friendly to LLVM's auto-vectorizer on the inner D-loop
    /// (per AGENTS.md hot-loop rule). No cross-lane reduction → bit-identical
    /// to scalar regardless of vectorization.
    #[inline]
    pub fn apply(&self, state: &mut [f32; D]) {
        for k in 0..K {
            let a = self.alphas[k];
            // Inner loop is the SAXPY `state[j] += a * block[k][j]`.
            // Use `mul_add` to match the FMA semantics of `simd_fused_scale_acc`
            // used by `apply_latent_steering` (Plan 309 parity, T1.8).
            let row = &self.block[k];
            for j in 0..D {
                state[j] = a.mul_add(row[j], state[j]);
            }
        }
    }

    /// The latent dimension `D`.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        D
    }

    /// The block size `K`.
    #[inline]
    #[must_use]
    pub const fn block_size(&self) -> usize {
        K
    }
}

// ── Free functions ──────────────────────────────────────────────────────────

/// Apply the field to a single latent state slice. Zero-alloc.
///
/// Thin wrapper over [`SubspaceSteeringField::apply`] for callers that hold a
/// reference (not the owned field). Computes
/// `state[j] += Σ_k alphas[k] * block[k][j]`.
#[inline]
pub fn apply_subspace_steering<const D: usize, const K: usize>(
    state: &mut [f32; D],
    field: &SubspaceSteeringField<D, K>,
) {
    field.apply(state);
}

/// Compute the BLAKE3 commitment over `block_le || alphas_le` (little-endian).
///
/// Deterministic, quorum-verifiable. Mirrors the `LatentSteeringVector`
/// commitment convention (`compute_commitment` in `latent_steering.rs`).
pub fn compute_block_commitment<const D: usize, const K: usize>(
    block: &[[f32; D]; K],
    alphas: &[f32; K],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for row in block.iter() {
        for &f in row.iter() {
            hasher.update(&f.to_le_bytes());
        }
    }
    for &a in alphas.iter() {
        hasher.update(&a.to_le_bytes());
    }
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

/// Per-axis projection energy `dot(block[k], state)` for `k in 0..K`.
///
/// Returns a fixed `[f32; K]` array where `out[k] = ⟨block[k], state⟩`. Used
/// for block-wise TopK consumption — which blocks are active in the current
/// state. Zero-alloc (output is a stack array).
///
/// This is the read-side counterpart of [`apply_subspace_steering`]: the
/// apply writes energy INTO the state (additive steering), this reads the
/// energy ALREADY PRESENT along each block axis (projection).
#[inline]
#[must_use]
pub fn block_energy<const D: usize, const K: usize>(
    block: &[[f32; D]; K],
    state: &[f32; D],
) -> [f32; K] {
    let mut out = [0f32; K];
    for k in 0..K {
        out[k] = dot_product(&block[k], state);
    }
    out
}

/// Sweep `alphas` over a grid and write the steered state at each grid point.
///
/// For each row `i` of `alpha_grid`, computes
/// `out_grid[i] = state + Σ_k alpha_grid[i][k] * block[k]` and writes it into
/// `out_grid[i]`. Zero-alloc after grid allocation (caller owns both grids).
///
/// This is the "pretzel manifold" pattern: each grid point is one concept
/// variation within the k-dim subspace. For `K=2`, a 2D `alpha_grid` sweeps a
/// surface; for `K=1`, a 1D grid sweeps a line.
///
/// # Panics
///
/// Panics (debug) if `alpha_grid.len() != out_grid.len()` or if
/// `alpha_grid[i].len() != K`. In release the lengths are trusted (the const
/// generic `K` makes the inner check a no-op when `alpha_grid` is `&[[f32; K]]`).
///
/// # Example
///
/// ```
/// use katgpt_core::subspace_steering::{walk_manifold, SubspaceSteeringField};
///
/// // K=1, D=8: sweep alpha over {0.0, 0.5, 1.0} → 3 steered states.
/// let mut block = [[0f32; 8]];
/// block[0][0] = 1.0;
/// let field = SubspaceSteeringField::<8, 1>::new_unchecked(block, [0.0]);
/// let state = [0f32; 8];
/// let alpha_grid = [[0.0f32], [0.5], [1.0]];
/// let mut out_grid = [[0f32; 8]; 3];
/// walk_manifold(&state, &field.block, &alpha_grid, &mut out_grid);
/// // out_grid[1] = state + 0.5 * block[0] = [0.5, 0, 0, 0, 0, 0, 0, 0]
/// assert_eq!(out_grid[1], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
pub fn walk_manifold<const D: usize, const K: usize>(
    state: &[f32; D],
    block: &[[f32; D]; K],
    alpha_grid: &[[f32; K]],
    out_grid: &mut [[f32; D]],
) {
    debug_assert_eq!(
        alpha_grid.len(),
        out_grid.len(),
        "alpha_grid and out_grid must have the same length"
    );
    for (alphas, out) in alpha_grid.iter().zip(out_grid.iter_mut()) {
        // Start from the base state.
        *out = *state;
        // Add Σ_k alphas[k] * block[k].
        for k in 0..K {
            let a = alphas[k];
            let row = &block[k];
            for (oj, &rj) in out.iter_mut().zip(row.iter()) {
                *oj += a * rj;
            }
        }
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

#[inline]
fn row_norm<const D: usize>(row: &[f32; D]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn dot_product<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a unit-norm 1D direction from a seed (deterministic, varied).
    fn make_unit_direction_1d(seed: u8, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|i| (seed as f32) * 0.1 + i as f32).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    /// Build a K-row orthonormal block from K seed directions via Gram-Schmidt
    /// (deterministic test fixture; production uses Newton-Schulz in T3.1).
    fn gram_schmidt_block<const D: usize, const K: usize>(seeds: [[u8; D]; K]) -> [[f32; D]; K] {
        let mut rows: Vec<[f32; D]> = (0..K).map(|_| [0f32; D]).collect();
        for (i, seed) in seeds.iter().enumerate() {
            // Raw direction from seed.
            let mut v: [f32; D] = [0f32; D];
            for (j, (vj, &sj)) in v.iter_mut().zip(seed.iter()).enumerate() {
                *vj = sj as f32 * 0.1 + (j as f32) + 1.0;
            }
            // Subtract projections onto already-orthonormalized rows.
            for prev in rows[..i].iter() {
                let proj = dot_product(&v, prev);
                for (vj, &pj) in v.iter_mut().zip(prev.iter()) {
                    *vj -= proj * pj;
                }
            }
            // Normalize.
            let n = row_norm(&v);
            for vj in v.iter_mut() {
                *vj /= n;
            }
            rows[i] = v;
        }
        // Copy into fixed array.
        let mut out = [[0f32; D]; K];
        out.copy_from_slice(&rows);
        out
    }

    /// `new` rejects non-unit-norm block rows.
    #[test]
    fn new_rejects_non_unit_norm() {
        let block = [[2.0f32; 8]]; // norm = sqrt(8*4) = 5.66, not 1.
        let alphas = [0.5f32];
        let err = SubspaceSteeringField::<8, 1>::new(block, alphas, 1e-5).unwrap_err();
        assert_eq!(err, SubspaceSteeringError::NotOrthonormal);
    }

    /// `new` rejects out-of-range alphas.
    #[test]
    fn new_rejects_alpha_out_of_range() {
        let block = [{
            let mut v = [0f32; 8];
            v[0] = 1.0;
            v
        }];
        let err = SubspaceSteeringField::<8, 1>::new(block, [1.5], 1e-5).unwrap_err();
        assert_eq!(err, SubspaceSteeringError::AlphaOutOfRange);
        let err = SubspaceSteeringField::<8, 1>::new(block, [-0.1], 1e-5).unwrap_err();
        assert_eq!(err, SubspaceSteeringError::AlphaOutOfRange);
    }

    /// `new` rejects non-orthogonal rows (K ≥ 2).
    #[test]
    fn new_rejects_non_orthogonal_rows() {
        // Two identical rows → dot = 1.0 (not orthogonal).
        let row = {
            let mut v = [0f32; 8];
            v[0] = 1.0;
            v
        };
        let block = [row, row];
        let err = SubspaceSteeringField::<8, 2>::new(block, [0.5, 0.5], 1e-5).unwrap_err();
        assert_eq!(err, SubspaceSteeringError::NotOrthonormal);
    }

    /// `verify` round-trips: construct → verify → true.
    #[test]
    fn verify_round_trip_passes() {
        let seeds = [[1u8; 8], [2u8; 8]];
        let block = gram_schmidt_block::<8, 2>(seeds);
        let field = SubspaceSteeringField::<8, 2>::new(block, [0.3, 0.7], 1e-4).unwrap();
        assert!(
            field.verify(1e-4),
            "verify must pass for a freshly-built field"
        );
    }

    /// `compute_block_commitment` is deterministic (same inputs → same hash).
    #[test]
    fn commitment_is_deterministic() {
        let seeds = [[3u8; 8], [4u8; 8]];
        let block = gram_schmidt_block::<8, 2>(seeds);
        let alphas = [0.2f32, 0.6];
        let h1 = compute_block_commitment(&block, &alphas);
        let h2 = compute_block_commitment(&block, &alphas);
        assert_eq!(h1, h2, "commitment must be deterministic");
        // And non-trivial (not all-zeros).
        assert_ne!(h1, [0u8; 32]);
    }

    /// Commitment is sensitive to a bit flip in the block or alphas.
    #[test]
    fn commitment_sensitive_to_tamper() {
        let seeds = [[5u8; 8], [6u8; 8]];
        let block = gram_schmidt_block::<8, 2>(seeds);
        let alphas = [0.2f32, 0.6];
        let h1 = compute_block_commitment(&block, &alphas);

        // Flip one element in block[0].
        let mut tampered_block = block;
        tampered_block[0][0] += 0.001;
        let h2 = compute_block_commitment(&tampered_block, &alphas);
        assert_ne!(h1, h2, "block tamper must change commitment");

        // Flip one alpha.
        let mut tampered_alphas = alphas;
        tampered_alphas[1] += 0.001;
        let h3 = compute_block_commitment(&block, &tampered_alphas);
        assert_ne!(h1, h3, "alpha tamper must change commitment");
    }

    /// `apply` shifts the state along the block's weighted sum direction.
    #[test]
    fn apply_shifts_state_along_weighted_block() {
        // K=1, D=8: state += alpha * block[0].
        let seeds = [[7u8; 8]];
        let block = gram_schmidt_block::<8, 1>(seeds);
        let field = SubspaceSteeringField::<8, 1>::new(block, [0.5], 1e-4).unwrap();

        let mut state = [0.1f32; 8];
        let original = state;
        field.apply(&mut state);

        // state[j] = original[j] + 0.5 * block[0][j].
        for j in 0..8 {
            let expected = original[j] + 0.5 * field.block[0][j];
            assert!(
                (state[j] - expected).abs() < 1e-6,
                "j={j}: got {}, expected {}",
                state[j],
                expected
            );
        }
    }

    /// **THE LOAD-BEARING GATE (T1.8):** `SubspaceSteeringField<D, 1>` is
    /// bit-identical to Plan 309's `apply_latent_steering`.
    ///
    /// Constructs a K=1 field from a Plan 309 direction + alpha, applies it to
    /// a test state, and asserts bit-identical output to the scalar reference.
    /// This proves the generalization subsumes the 1D case.
    #[test]
    fn k1_parity_with_plan_309() {
        // Plan 309 ships DEFAULT-ON, so it's always available.
        use crate::latent_steering::{LatentSteeringVector, apply_latent_steering};

        const D: usize = 8;
        let direction_v = make_unit_direction_1d(42, D);
        let alpha = 0.3f32;

        // Plan 309 reference.
        let steering = LatentSteeringVector::new_unchecked(direction_v.clone(), alpha);

        // Plan 412 K=1 field from the same direction + alpha.
        let mut block = [[0f32; D]];
        block[0].copy_from_slice(&direction_v);
        let field = SubspaceSteeringField::<D, 1>::new(block, [alpha], 1e-5).unwrap();

        // Identical starting state.
        let state_plan309 = [0.1f32, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4];
        let mut s_ref = state_plan309;
        let mut s_412 = state_plan309;

        apply_latent_steering(&mut s_ref, &steering);
        field.apply(&mut s_412);

        // Bit-identical: same operation (state += alpha * direction), same
        // element-wise rounding.
        for j in 0..D {
            assert_eq!(
                s_ref[j].to_bits(),
                s_412[j].to_bits(),
                "K=1 parity FAIL at j={j}: plan309={} plan412={}",
                s_ref[j],
                s_412[j]
            );
        }
    }

    /// K=2 field: `apply` is the sum of two weighted orthonormal directions.
    /// Each axis shifts independently (orthonormality means no cross-term).
    #[test]
    fn k2_apply_sums_two_orthonormal_axes() {
        const D: usize = 8;
        let seeds = [[1u8; D], [2u8; D]];
        let block = gram_schmidt_block::<D, 2>(seeds);
        let alphas = [0.4f32, 0.6];
        let field = SubspaceSteeringField::<D, 2>::new(block, alphas, 1e-4).unwrap();

        let mut state = [0.0f32; D];
        field.apply(&mut state);

        // state[j] = 0.4 * block[0][j] + 0.6 * block[1][j] (starting from 0).
        for (j, &sj) in state.iter().enumerate() {
            let expected = alphas[0] * field.block[0][j] + alphas[1] * field.block[1][j];
            assert!(
                (sj - expected).abs() < 1e-6,
                "j={j}: got {sj}, expected {expected}"
            );
        }
    }

    /// `new` with K=0 degenerate case: no rows, no alphas. The field is a
    /// no-op (apply does nothing). Commitment is BLAKE3 of empty input.
    #[test]
    fn k0_field_is_noop() {
        let field = SubspaceSteeringField::<8, 0>::new([], [], 1e-5).unwrap();
        let mut state = [0.5f32; 8];
        let original = state;
        field.apply(&mut state);
        assert_eq!(state, original, "K=0 field must be a no-op");
        // Commitment is still well-defined (BLAKE3 of empty).
        assert_ne!(field.commitment, [0u8; 32]);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Plan 412 Phase 2 — block_energy + walk_manifold tests
    // ──────────────────────────────────────────────────────────────────────

    /// `block_energy` returns the per-axis dot projection of state onto each
    /// block row.
    #[test]
    fn block_energy_returns_per_axis_projection() {
        const D: usize = 8;
        // Two orthonormal axes: e_0 and e_1 (standard basis).
        let mut block = [[0f32; D]; 2];
        block[0][0] = 1.0;
        block[1][1] = 1.0;
        let state = [0.3f32, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let energy = block_energy(&block, &state);
        assert!(
            (energy[0] - 0.3).abs() < 1e-6,
            "axis 0 energy = dot(e_0, state) = 0.3"
        );
        assert!(
            (energy[1] - 0.7).abs() < 1e-6,
            "axis 1 energy = dot(e_1, state) = 0.7"
        );
    }

    /// `block_energy` on a state aligned with block[0] returns high energy on
    /// axis 0 and ~0 on the orthogonal axes.
    #[test]
    fn block_energy_aligned_state_dominates_one_axis() {
        const D: usize = 8;
        // Use standard basis vectors e_0, e_1, e_2 — a trivially orthonormal
        // block (no Gram-Schmidt numerical-conditioning concerns).
        let mut block = [[0f32; D]; 3];
        block[0][0] = 1.0;
        block[1][1] = 1.0;
        block[2][2] = 1.0;
        // State = exactly block[0] (perfectly aligned with axis 0).
        let state = block[0];
        let energy = block_energy(&block, &state);
        // Energy on axis 0 = dot(block[0], block[0]) = 1.0 (unit norm).
        assert!(
            (energy[0] - 1.0).abs() < 1e-6,
            "aligned axis energy ~ 1.0, got {}",
            energy[0]
        );
        // Energy on orthogonal axes ~ 0 (by orthonormality).
        assert!(
            energy[1].abs() < 1e-6,
            "orthogonal axis 1 energy ~ 0, got {}",
            energy[1]
        );
        assert!(
            energy[2].abs() < 1e-6,
            "orthogonal axis 2 energy ~ 0, got {}",
            energy[2]
        );
    }

    /// `walk_manifold` produces steered states at each grid point.
    /// K=1: sweep alpha over a line.
    #[test]
    fn walk_manifold_k1_sweeps_a_line() {
        const D: usize = 8;
        let mut block = [[0f32; D]];
        block[0][0] = 1.0; // e_0
        let field = SubspaceSteeringField::<D, 1>::new(block, [0.0], 1e-5).unwrap();
        let state = [0.5f32; D];
        let alpha_grid = [[0.0f32], [0.5], [1.0], [2.0]];
        let mut out_grid = [[0f32; D]; 4];
        walk_manifold(&state, &field.block, &alpha_grid, &mut out_grid);

        // out_grid[i] = state + alpha_grid[i][0] * e_0
        for i in 0..4 {
            let expected_first = state[0] + alpha_grid[i][0];
            assert!((out_grid[i][0] - expected_first).abs() < 1e-6);
            // Other components unchanged.
            for j in 1..D {
                assert!((out_grid[i][j] - state[j]).abs() < 1e-6);
            }
        }
    }

    /// **T2.3:** K=2 walk preserves norm bounds — each walked output's L2 norm
    /// is within `[‖state‖ − ε, ‖state‖ + Σ_k |α_k|]`. Per Plan 322's
    /// norm-preservation analysis, the norm inflation from additive steering
    /// is bounded by the steering magnitude (triangle inequality).
    #[test]
    fn k2_walk_preserves_norm_bounds() {
        const D: usize = 8;
        // Distinct per-element seeds for well-conditioned Gram-Schmidt.
        let seeds = [[1u8, 2, 3, 4, 5, 6, 7, 8], [8u8, 7, 6, 5, 4, 3, 2, 1]];
        let block = gram_schmidt_block::<D, 2>(seeds);
        let field = SubspaceSteeringField::<D, 2>::new(block, [0.0, 0.0], 1e-4).unwrap();

        let state = [0.1f32, 0.2, 0.1, 0.05, 0.15, 0.1, 0.0, 0.08];
        let state_norm = row_norm(&state);

        // Sweep a 5x5 grid over [-0.5, 0.5]^2.
        let mut alpha_grid: Vec<[f32; 2]> = Vec::new();
        for a0 in [-0.5f32, -0.25, 0.0, 0.25, 0.5] {
            for a1 in [-0.5f32, -0.25, 0.0, 0.25, 0.5] {
                alpha_grid.push([a0, a1]);
            }
        }
        let mut out_grid = vec![[0f32; D]; alpha_grid.len()];
        walk_manifold(&state, &field.block, &alpha_grid, &mut out_grid);

        for (i, out) in out_grid.iter().enumerate() {
            let out_norm = row_norm(out);
            let sum_abs_alpha = alpha_grid[i].iter().map(|a| a.abs()).sum::<f32>();
            // The steering adds a vector of norm <= Σ_k |α_k| (each block row
            // is unit-norm, so ||Σ α_k u_k|| <= Σ |α_k|). By triangle inequality:
            //   |out_norm - state_norm| <= ||Σ α_k u_k|| <= Σ |α_k|.
            let lower = state_norm - sum_abs_alpha - 1e-5;
            let upper = state_norm + sum_abs_alpha + 1e-5;
            assert!(
                out_norm >= lower && out_norm <= upper,
                "T2.3 norm bound FAIL at grid i={i}: out_norm={out_norm}, state_norm={state_norm}, sum_abs_alpha={sum_abs_alpha}, bounds=[{lower}, {upper}]"
            );
        }
    }

    /// **T2.4:** K=2 walk covers the grid — the walked grid produces distinct
    /// output states (no duplicates unless alphas repeat).
    #[test]
    fn k2_walk_covers_grid() {
        const D: usize = 8;
        // Distinct per-element seeds for well-conditioned Gram-Schmidt.
        let seeds = [[1u8, 2, 3, 4, 5, 6, 7, 8], [8u8, 7, 6, 5, 4, 3, 2, 1]];
        let block = gram_schmidt_block::<D, 2>(seeds);
        let field = SubspaceSteeringField::<D, 2>::new(block, [0.0, 0.0], 1e-4).unwrap();

        let state = [0.0f32; D];
        // 4 distinct alpha pairs → 4 distinct output states (block rows are
        // linearly independent, so different alpha pairs give different sums).
        let alpha_grid = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut out_grid = [[0f32; D]; 4];
        walk_manifold(&state, &field.block, &alpha_grid, &mut out_grid);

        // All 4 outputs must be distinct (different alpha pairs → different
        // linear combinations of two linearly-independent block rows).
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(
                    out_grid[i], out_grid[j],
                    "T2.4 grid coverage FAIL: out_grid[{i}] == out_grid[{j}] for distinct alphas"
                );
            }
        }

        // Repeated alpha pair → repeated output (determinism).
        let alpha_repeat = [[0.5f32, 0.5], [0.5, 0.5]];
        let mut out_repeat = [[0f32; D]; 2];
        walk_manifold(&state, &field.block, &alpha_repeat, &mut out_repeat);
        assert_eq!(
            out_repeat[0], out_repeat[1],
            "repeated alphas must give identical outputs"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Plan 412 Phase 3 — from_directions_orthonormalize tests
    // ──────────────────────────────────────────────────────────────────────

    /// **T3.2:** `from_directions_orthonormalize` produces an orthonormal block
    /// from a set of nearly-orthogonal directions via Newton-Schulz.
    #[test]
    fn from_directions_orthonormalize_produces_orthonormal_block() {
        const D: usize = 8;
        const K: usize = 3;
        // Start from standard basis vectors e_0, e_1, e_2 (already perfectly
        // orthonormal). Newton-Schulz + row renormalization should return them
        // orthonormal (the rows stay orthogonal; renormalization fixes the
        // unit-norm property that NS's 5-iter cubic underconverges on
        // non-square K<D matrices).
        let mut directions = [[0f32; D]; K];
        directions[0][0] = 1.0;
        directions[1][1] = 1.0;
        directions[2][2] = 1.0;
        let field = SubspaceSteeringField::<D, K>::from_directions_orthonormalize(
            &directions,
            [0.3, 0.5, 0.7],
            1e-4,
        )
        .expect("orthonormalize on already-orthonormal input must succeed");

        // Output block must be orthonormal.
        assert!(field.verify(1e-4), "output block must be orthonormal");
        // Pairwise dots must be ~0 (orthogonality preserved by NS).
        for i in 0..K {
            for j in (i + 1)..K {
                let dot = dot_product(&field.block[i], &field.block[j]);
                assert!(
                    dot.abs() < 1e-4,
                    "block[{i}]·block[{j}]={dot} must be < 1e-4 (NS orthogonality)"
                );
            }
        }
        // Each row must be unit-norm (renormalization post-pass).
        for k in 0..K {
            let n = row_norm(&field.block[k]);
            assert!(
                (n - 1.0).abs() < 1e-4,
                "block[{k}] norm {n} must be ~1.0 (renormalized)"
            );
        }
    }

    /// `from_directions_orthonormalize` orthogonalizes a set of arbitrary
    /// (possibly far-from-orthogonal) directions into an orthonormal block
    /// via Gram-Schmidt.
    #[test]
    fn from_directions_orthonormalize_cleans_up_drift() {
        const D: usize = 8;
        const K: usize = 2;
        // Two far-from-orthogonal directions (dot = 2.0, angle ~60°).
        // Gram-Schmidt is exact for ANY input (unlike Newton-Schulz, which
        // diverges on non-square K<D matrices — see Plan 412 Phase 3 finding).
        let mut directions = [[0f32; D]; K];
        directions[0] = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        directions[1] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // The raw input is far from orthogonal.
        let raw_dot = dot_product(&directions[0], &directions[1]);
        assert!(
            raw_dot.abs() > 0.5,
            "fixture guard: raw input must be far from orthogonal"
        );

        // Gram-Schmidt is exact — tight tolerance is appropriate.
        let field = SubspaceSteeringField::<D, K>::from_directions_orthonormalize(
            &directions,
            [0.4, 0.6],
            1e-4,
        )
        .expect("Gram-Schmidt must orthogonalize arbitrary input exactly");

        // Output block must be orthonormal.
        assert!(
            field.verify(1e-4),
            "orthogonalized block must be orthonormal"
        );
        // The dot product must be ~0 (Gram-Schmidt is exact).
        let clean_dot = dot_product(&field.block[0], &field.block[1]);
        assert!(
            clean_dot.abs() < 1e-4,
            "orthogonalized block rows must be orthogonal, got dot = {clean_dot}"
        );
    }

    /// `from_directions_orthonormalize` propagates the alpha-range check
    /// (rejects out-of-range alphas even before orthogonalizing).
    #[test]
    fn from_directions_orthonormalize_rejects_bad_alphas() {
        const D: usize = 8;
        let mut directions = [[0f32; D]; 2];
        directions[0][0] = 1.0;
        directions[1][1] = 1.0;
        let err = SubspaceSteeringField::<D, 2>::from_directions_orthonormalize(
            &directions,
            [0.5, 1.5], // 1.5 out of range
            1e-4,
        )
        .unwrap_err();
        assert_eq!(err, SubspaceSteeringError::AlphaOutOfRange);
    }
}
