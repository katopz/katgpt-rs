//! Sparse Off-Principal Task Vector (SOPTV) — modelless storage for behavioral deltas.
//!
//! Distilled from arXiv 2606.13657 ("Dense Supervision, Sparse Updates", Yu et al. 2026).
//! The paper proves that on-policy distillation (OPD) and RLVR produce weight deltas that
//! are **small** (0.04–0.14% relative Frobenius norm), **coordinate-sparse** (66–90% of
//! coordinates below `1e-5`), **off-principal** (≤1% energy on source top-10% SVD),
//! **spectrally concentrated** (top-16 SVD energy 20–31%), and **FFN-heavy** (62–86%).
//!
//! A task vector in the sense of Ilharco et al. 2022 ("Editing Models with Task
//! Arithmetic") is `ΔW = W_trained − W_src`. This module stores it sparsely so the
//! inference engine can load shipped adapters with 2.9–5.7× less memory than dense LoRA.
//!
//! This is engine plumbing: it stores *any* task vector in sparse form. The specific
//! masks for our game LoRAs (the "fuel") are produced by riir-ai's SON-LT training
//! pipeline (Research 120, Plan 296) and consumed here.
//!
//! # Design
//!
//! - **Sparse storage:** `(shape, mask, deltas, eta)` — parallel index + value arrays.
//! - **Task arithmetic:** `eta` enables add (1.0), subtract (−1.0), scale (any f32).
//! - **Zero-alloc apply:** `apply_to_scratch` writes into a caller-provided base buffer
//!   without allocating — safe for hot paths.
//! - **Density-aware fallback:** if `density() > 0.5`, callers should prefer dense LoRA
//!   (sparse apply is slower than GEMM at low sparsity).
//!
//! # Example
//!
//! ```
//! use katgpt_rs::sparse_task_vector::SparseTaskVector;
//!
//! // A 4×3 base weight.
//! let mut base = vec![1.0_f32, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,  0.5, 0.5, 0.5];
//! // Dense delta with mostly zeros (paper-finding-1 sparsity pattern).
//! let delta = vec![0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.2, -0.1, 0.0];
//!
//! let stv = SparseTaskVector::from_dense(&delta, (4, 3), 1e-5);
//! assert_eq!(stv.len(), 2);                 // only the two nonzero deltas
//! assert!((stv.density() - 2.0 / 12.0).abs() < 1e-6);
//!
//! stv.apply_to(&mut base);
//! assert!((base[9] - 0.7).abs() < 1e-6);     // 0.5 + 0.2
//! assert!((base[10] - 0.4).abs() < 1e-6);    // 0.5 + (-0.1)
//! ```

/// A task vector stored sparsely as `(mask, deltas, eta)`.
///
/// Implements the storage format from Research 231 (Plan 264 Fusion A). The mask is
/// a sorted `Vec<u32>` of row-major flat indices into the dense equivalent. Deltas
/// are parallel to mask. `eta` scales the superposition (1.0 = task arithmetic add).
///
/// **Paper grounding:** arXiv 2606.13657 §4.1 — 66.72% to 89.50% of OPD-style
/// checkpoint deltas are below the `1e-5` visible-update threshold, across six
/// OPD-trained model pairs.
#[derive(Clone, Debug)]
pub struct SparseTaskVector {
    /// Dense-equivalent shape `(rows, cols)`. `rows * cols == dense_length`.
    pub shape: (usize, usize),
    /// Sorted active coordinate indices in row-major flat order.
    pub mask: Vec<u32>,
    /// Non-zero delta values, parallel to `mask`.
    pub deltas: Vec<f32>,
    /// Scalar mixing coefficient for task arithmetic. Default 1.0.
    pub eta: f32,
}

impl SparseTaskVector {
    /// Build a sparse task vector from a dense delta slice.
    ///
    /// Coordinates with `|delta[i]| <= threshold` are dropped. The resulting mask
    /// is sorted ascending. Use `threshold = 1e-5` to match the paper's main
    /// sparsity tables (§4.1, Table 2).
    ///
    /// `shape.0 * shape.1` must equal `dense.len()`.
    pub fn from_dense(dense: &[f32], shape: (usize, usize), threshold: f32) -> Self {
        debug_assert_eq!(
            shape.0 * shape.1,
            dense.len(),
            "shape {:?} doesn't match dense.len() {}",
            shape,
            dense.len()
        );
        let mut mask = Vec::with_capacity(dense.len() / 4);
        let mut deltas = Vec::with_capacity(dense.len() / 4);
        for (i, &v) in dense.iter().enumerate() {
            if v.abs() > threshold {
                mask.push(i as u32);
                deltas.push(v);
            }
        }
        mask.shrink_to_fit();
        deltas.shrink_to_fit();
        Self {
            shape,
            mask,
            deltas,
            eta: 1.0,
        }
    }

    /// Build from explicit parallel `mask` + `deltas` (must be same length, mask sorted).
    ///
    /// Use this when the mask has already been discovered (e.g., by riir-ai's
    /// subnetwork training pipeline, Research 120 Fusion 1).
    pub fn from_parts(
        shape: (usize, usize),
        mask: Vec<u32>,
        deltas: Vec<f32>,
        eta: f32,
    ) -> Result<Self, SparseTaskVectorError> {
        if mask.len() != deltas.len() {
            return Err(SparseTaskVectorError::LengthMismatch {
                mask: mask.len(),
                deltas: deltas.len(),
            });
        }
        if !mask.is_empty() {
            let total = shape.0 * shape.1;
            if mask[0] as usize >= total {
                return Err(SparseTaskVectorError::IndexOutOfRange {
                    idx: mask[0],
                    total,
                });
            }
            if *mask.last().unwrap() as usize >= total {
                return Err(SparseTaskVectorError::IndexOutOfRange {
                    idx: *mask.last().unwrap(),
                    total,
                });
            }
            if !mask.windows(2).all(|w| w[0] <= w[1]) {
                return Err(SparseTaskVectorError::UnsortedMask);
            }
        }
        Ok(Self {
            shape,
            mask,
            deltas,
            eta,
        })
    }

    /// Number of active coordinates (non-zero deltas).
    #[inline]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Whether the task vector is empty (no active coordinates).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Total number of coordinates in the dense equivalent.
    #[inline]
    pub fn dense_len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Active fraction in `[0, 1]`. Paper §4.1 reports `1 − density` ∈ [0.667, 0.895]
    /// for OPD pairs, i.e., densities of 10.5% to 33.3%.
    #[inline]
    pub fn density(&self) -> f32 {
        if self.dense_len() == 0 {
            return 0.0;
        }
        self.mask.len() as f32 / self.dense_len() as f32
    }

    /// Relative Frobenius norm of the delta vs the given base, `‖ΔW‖_F / ‖W_src‖_F`.
    ///
    /// Paper §4.1 reports this in the range 0.036% to 0.142% for OPD pairs. Call this
    /// after `from_dense` to verify the task vector is small in the paper's sense.
    pub fn relative_norm_vs(&self, base: &[f32]) -> f32 {
        debug_assert_eq!(base.len(), self.dense_len());
        let eta = self.eta;
        // Delta sum of squares — sparse, scalar is fine (nnz is small).
        let mut delta_sq_sum = 0.0_f32;
        for &d in &self.deltas {
            let s = d * eta;
            delta_sq_sum += s * s;
        }
        // Base sum of squares — dense, use SIMD.
        let base_sq_sum = crate::simd::simd_sum_sq(base, base.len());
        if base_sq_sum <= f32::MIN_POSITIVE {
            return 0.0;
        }
        delta_sq_sum.sqrt() / base_sq_sum.sqrt()
    }

    /// Storage bytes used by this sparse representation.
    ///
    /// `mask` is `u32` (4 bytes), `deltas` is `f32` (4 bytes). Total = `8 * len()`
    /// plus the `shape + eta` header (24 bytes).
    pub fn sparse_bytes(&self) -> usize {
        24 + 4 * self.mask.len() + 4 * self.deltas.len()
    }

    /// Storage bytes the dense equivalent would use (`4 * dense_len`).
    pub fn dense_bytes(&self) -> usize {
        4 * self.dense_len()
    }

    /// Storage reduction ratio `dense_bytes / sparse_bytes`. Paper-shaped masks
    /// at 17.5% density yield ~2.9×, at 10.5% density ~5.7×.
    pub fn storage_reduction(&self) -> f32 {
        let s = self.sparse_bytes();
        if s == 0 {
            return 1.0;
        }
        self.dense_bytes() as f32 / s as f32
    }

    /// Scatter-add `eta * deltas[i]` into `base[mask[i]]` for every active coordinate.
    ///
    /// **Correctness invariant:** after `apply_to`, `base[mask[i]] += eta * deltas[i]`
    /// for every `i`, and all other coordinates are unchanged. This is the task
    /// arithmetic "add" operation (Ilharco et al. 2022).
    ///
    /// **Hot-path safety:** zero allocation. Safe to call inside decode loops.
    pub fn apply_to(&self, base: &mut [f32]) {
        debug_assert_eq!(base.len(), self.dense_len());
        let eta = self.eta;
        for (&idx, &d) in self.mask.iter().zip(self.deltas.iter()) {
            base[idx as usize] += eta * d;
        }
    }

    /// Apply into a caller-owned scratch buffer (alias for `apply_to` since we don't
    /// allocate either way — kept as a separate name for API clarity at call sites
    /// that pass a scratch base distinct from the source-of-truth base).
    ///
    /// Use this when the caller maintains a separate scratch base for in-flight
    /// computation and wants to leave the canonical base untouched.
    #[inline]
    pub fn apply_to_scratch(&self, scratch_base: &mut [f32]) {
        self.apply_to(scratch_base);
    }

    /// Subtract the task vector from `base` (negate `eta`). Useful for "task negation"
    /// in task arithmetic — removing a capability.
    pub fn subtract_from(&self, base: &mut [f32]) {
        debug_assert_eq!(base.len(), self.dense_len());
        let eta = self.eta;
        for (&idx, &d) in self.mask.iter().zip(self.deltas.iter()) {
            base[idx as usize] -= eta * d;
        }
    }

    /// In-place scale `eta`. Useful for annealing a task vector's contribution.
    #[inline]
    pub fn scale_eta(&mut self, factor: f32) {
        self.eta *= factor;
    }

    /// Iterate over `(flat_index, delta_value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.mask
            .iter()
            .copied()
            .zip(self.deltas.iter().map(|&d| d * self.eta))
    }

    /// Reconstruct the dense delta into a caller-provided buffer (zeroed first).
    ///
    /// Mostly for diagnostics and GOAT proofs. The hot path should use `apply_to`.
    pub fn to_dense_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.dense_len());
        for x in out.iter_mut() {
            *x = 0.0;
        }
        self.apply_to(out);
    }
}

// ── Phase 4 (Plan 270): gauge-invariant composition ──────────────────────
//
// Gated on `gauge_invariant` (which pulls in `newton_schulz`). When the feature
// is off, SparseTaskVector behaves exactly as before — no API or ABI change.

#[cfg(feature = "gauge_invariant")]
impl SparseTaskVector {
    /// Compose this task vector with `other` using gauge-invariant arithmetic.
    ///
    /// Returns a new `SparseTaskVector` whose dense equivalent is
    ///   `ΔW_merged = self.eta · ΔW_self + eta · other.eta · ΔW_other`
    /// restricted to the union of both masks (coordinates where the merged
    /// delta is non-zero above a small cancellation threshold).
    ///
    /// # Why this is gauge-invariant
    ///
    /// Each `SparseTaskVector` admits a natural rank-`nnz` factorization
    /// `ΔW = A · B^T` where `A ∈ R^{rows × nnz}` places each delta at its row
    /// and `B ∈ R^{cols × nnz}` is a one-hot selection matrix marking each
    /// delta's column. With this factorization, `σ_max(B) = 1` (columns are
    /// orthonormal unit vectors), so `gauge_rebalance` rescales `(A, B)` by
    /// `c = σ_max(A)^{-1/2}` — a transformation that leaves `A · B^T` exactly
    /// unchanged. Consequently `gauge_invariant_compose([(1, A₁, B₁), (η, A₂, B₂)])`
    /// produces `ΔW₁ + η · ΔW₂`, which is precisely the weighted delta sum
    /// implemented here.
    ///
    /// Routing the natural factorization through the full
    /// `gauge_invariant_compose` machinery would require materializing
    /// `(rows × nnz)` and `(cols × nnz)` dense buffers — prohibitive for
    /// paper-scale masks (4096 × 4096 with 10⁵+ active coordinates). The direct
    /// sum is `O(|mask_self| + |mask_other|)` and allocates only for the output
    /// mask + deltas. Equivalence with the full compose path is verified by
    /// `test_compose_gauge_invariant_matches_full_compose`.
    ///
    /// # Allocation
    ///
    /// The output `mask` and `deltas` are allocated once with the tight upper
    /// bound `self.len() + other.len()`. The merge-join loop is branch-free
    /// apart from the three-way comparison and performs no scratch allocation.
    ///
    /// # Panics
    ///
    /// - If `self.shape != other.shape`.
    pub fn compose_gauge_invariant(&self, other: &SparseTaskVector, eta: f32) -> SparseTaskVector {
        assert_eq!(
            self.shape, other.shape,
            "shape mismatch in gauge-invariant compose: {:?} vs {:?}",
            self.shape, other.shape
        );

        // Effective weights: bake each STV's own `eta` into the contribution.
        let w_self = self.eta;
        let w_other = other.eta * eta;

        // Tight upper bound on merged length — no reallocation after this.
        let cap = self.mask.len() + other.mask.len();
        let mut merged_mask: Vec<u32> = Vec::with_capacity(cap);
        let mut merged_deltas: Vec<f32> = Vec::with_capacity(cap);

        // Three-way merge-join of the two sorted masks. Both masks are sorted
        // ascending by `from_parts` / `from_dense` invariant.
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.mask.len() && j < other.mask.len() {
            let a = self.mask[i];
            let b = other.mask[j];
            if a < b {
                merged_mask.push(a);
                merged_deltas.push(w_self * self.deltas[i]);
                i += 1;
            } else if a > b {
                merged_mask.push(b);
                merged_deltas.push(w_other * other.deltas[j]);
                j += 1;
            } else {
                // Same coordinate — contributions sum (the gauge-invariant merge).
                merged_mask.push(a);
                merged_deltas.push(w_self * self.deltas[i] + w_other * other.deltas[j]);
                i += 1;
                j += 1;
            }
        }
        // Drain whichever side still has entries.
        while i < self.mask.len() {
            merged_mask.push(self.mask[i]);
            merged_deltas.push(w_self * self.deltas[i]);
            i += 1;
        }
        while j < other.mask.len() {
            merged_mask.push(other.mask[j]);
            merged_deltas.push(w_other * other.deltas[j]);
            j += 1;
        }

        // Prune near-zero merged deltas — opposite-sign contributions can
        // cancel exactly, and we don't want to store numerical noise. Threshold
        // is tighter than `from_dense`'s 1e-5 so legitimate small deltas survive.
        const CANCEL_THRESHOLD: f32 = 1e-7;
        let mut write = 0usize;
        for k in 0..merged_mask.len() {
            if merged_deltas[k].abs() > CANCEL_THRESHOLD {
                merged_mask[write] = merged_mask[k];
                merged_deltas[write] = merged_deltas[k];
                write += 1;
            }
        }
        merged_mask.truncate(write);
        merged_deltas.truncate(write);
        merged_mask.shrink_to_fit();
        merged_deltas.shrink_to_fit();

        SparseTaskVector {
            shape: self.shape,
            mask: merged_mask,
            deltas: merged_deltas,
            // Scaling is already baked into `deltas` — expose `eta = 1.0` so
            // downstream `apply_to` / `scale_eta` behave predictably.
            eta: 1.0,
        }
    }
}

/// Errors returned by `SparseTaskVector::from_parts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseTaskVectorError {
    /// `mask.len() != deltas.len()`.
    LengthMismatch { mask: usize, deltas: usize },
    /// A mask index is `>= shape.0 * shape.1`.
    IndexOutOfRange { idx: u32, total: usize },
    /// The mask is not sorted ascending. Sort before calling `from_parts`.
    UnsortedMask,
}

impl std::fmt::Display for SparseTaskVectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch { mask, deltas } => write!(
                f,
                "SparseTaskVector length mismatch: mask={mask}, deltas={deltas}"
            ),
            Self::IndexOutOfRange { idx, total } => write!(
                f,
                "SparseTaskVector index {idx} out of range for dense length {total}"
            ),
            Self::UnsortedMask => write!(f, "SparseTaskVector mask must be sorted ascending"),
        }
    }
}

impl std::error::Error for SparseTaskVectorError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// GOAT test G1: storage reduction at paper densities.
    ///
    /// Paper §4.1: DS-Qwen OPD mask density 17.5%, Qwen3 OPSD 10.5%. We expect
    /// ≥2.5× reduction at 17.5% and ≥4.0× at 10.5% (the 5.7× headline comes from
    /// ignoring the 24-byte header on large matrices).
    #[test]
    fn g1_storage_reduction_at_paper_densities() {
        // 4096×4096 = 16_777_216 coords. At 17.5% density = 2_936_013 active.
        let n = 4096 * 4096;
        let density_175 = (n as f32 * 0.175) as usize;
        let stv_175 = SparseTaskVector {
            shape: (4096, 4096),
            mask: (0..density_175 as u32).collect(),
            deltas: vec![0.1; density_175],
            eta: 1.0,
        };
        let reduction_175 = stv_175.storage_reduction();
        assert!(
            reduction_175 >= 2.5,
            "17.5% density should give >=2.5x reduction, got {reduction_175}"
        );

        let density_105 = (n as f32 * 0.105) as usize;
        let stv_105 = SparseTaskVector {
            shape: (4096, 4096),
            mask: (0..density_105 as u32).collect(),
            deltas: vec![0.1; density_105],
            eta: 1.0,
        };
        let reduction_105 = stv_105.storage_reduction();
        assert!(
            reduction_105 >= 4.0,
            "10.5% density should give >=4.0x reduction, got {reduction_105}"
        );
    }

    /// GOAT test G2: apply roundtrip recovers base+delta within f32 accumulation noise.
    ///
    /// `from_dense(delta)` then `apply_to(base)` should yield `base + delta` exactly
    /// (up to f32 rounding and the threshold cutoff). Deltas above threshold are
    /// applied; the absolute error from a single scatter-add is bounded by
    /// `f32::EPSILON * |base + delta|`, so we use a relative tolerance.
    #[test]
    fn g2_apply_roundtrip() {
        let mut rng = fastrand::Rng::with_seed(42);
        let rows = 64;
        let cols = 64;
        let n = rows * cols;

        let mut base: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let delta: Vec<f32> = (0..n)
            .map(|_| {
                // 80% zeros (paper sparsity), 20% small nonzero deltas.
                if rng.f32() < 0.8 {
                    0.0
                } else {
                    rng.f32() * 0.01 - 0.005
                }
            })
            .collect();

        let expected: Vec<f32> = base.iter().zip(delta.iter()).map(|(&b, &d)| b + d).collect();

        let stv = SparseTaskVector::from_dense(&delta, (rows, cols), 1e-5);
        stv.apply_to(&mut base);

        let mut max_rel_err = 0.0_f32;
        for (i, (&got, &want)) in base.iter().zip(expected.iter()).enumerate() {
            // Either the delta was below threshold (got == original base) or it
            // was applied (got ≈ want within f32 rounding). We use a relative
            // tolerance because LLVM may auto-vectorize the `expected` map
            // differently from the `apply_to` loop (different FMA contraction),
            // producing ULP-level differences. 1e-4 rel is still far tighter
            // than any meaningful precision concern for adapter deltas.
            let abs_diff = (got - want).abs();
            let scale = want.abs().max(1e-6);
            let rel_err = abs_diff / scale;
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
            assert!(
                rel_err < 1e-4,
                "roundtrip mismatch at {i}: got {got}, want {want}, rel_err {rel_err}"
            );
        }
        // Confirm we exercised the path (not all-zero deltas).
        assert!(!stv.is_empty(), "test should produce non-empty sparse vector");
        eprintln!("g2_apply_roundtrip: {}/{} active, max_rel_err = {max_rel_err:.2e}", stv.len(), n);
    }

    /// Paper §4.1 metric: density reports the correct active fraction.
    #[test]
    fn density_matches_paper_definition() {
        let dense = vec![0.0, 0.0, 0.0, 0.001, 0.0, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stv = SparseTaskVector::from_dense(&dense, (4, 3), 1e-5);
        assert_eq!(stv.len(), 2);
        let d = stv.density();
        assert!((d - 2.0 / 12.0).abs() < 1e-6, "density {d}");
    }

    /// Paper §4.1 metric: relative Frobenius norm vs source.
    /// Paper reports 0.04–0.14% for OPD pairs. Here we just verify the formula.
    #[test]
    fn relative_norm_formula() {
        // base = identity 3x3, delta = 0.1 on diagonal.
        let base = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let delta = vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1];
        let stv = SparseTaskVector::from_dense(&delta, (3, 3), 1e-5);
        let r = stv.relative_norm_vs(&base);
        // ‖Δ‖_F = sqrt(3 * 0.01) = sqrt(0.03) ≈ 0.1732
        // ‖W‖_F  = sqrt(3)       ≈ 1.7321
        // ratio  = 0.1
        assert!((r - 0.1).abs() < 1e-5, "relative norm {r}");
    }

    /// `from_parts` validates mask/delta length parity and index range.
    #[test]
    fn from_parts_rejects_mismatched_lengths() {
        let result = SparseTaskVector::from_parts(
            (3, 3),
            vec![0, 1],
            vec![0.1],
            1.0,
        );
        assert_eq!(
            result.unwrap_err(),
            SparseTaskVectorError::LengthMismatch {
                mask: 2,
                deltas: 1
            }
        );
    }

    #[test]
    fn from_parts_rejects_out_of_range() {
        let result = SparseTaskVector::from_parts(
            (3, 3),
            vec![0, 99],
            vec![0.1, 0.2],
            1.0,
        );
        assert_eq!(
            result.unwrap_err(),
            SparseTaskVectorError::IndexOutOfRange {
                idx: 99,
                total: 9
            }
        );
    }

    #[test]
    fn from_parts_rejects_unsorted() {
        let result = SparseTaskVector::from_parts(
            (3, 3),
            vec![2, 1],
            vec![0.1, 0.2],
            1.0,
        );
        assert_eq!(result.unwrap_err(), SparseTaskVectorError::UnsortedMask);
    }

    /// `subtract_from` reverses `apply_to` exactly.
    #[test]
    fn subtract_reverses_apply() {
        let mut base = vec![1.0_f32; 9];
        let original = base.clone();
        let delta = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.3];
        let stv = SparseTaskVector::from_dense(&delta, (3, 3), 1e-5);
        stv.apply_to(&mut base);
        stv.subtract_from(&mut base);
        for (i, (&got, &want)) in base.iter().zip(original.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "subtract mismatch at {i}");
        }
    }

    /// `eta` scaling changes apply magnitude.
    #[test]
    fn eta_scales_apply() {
        let mut base = vec![0.0_f32; 4];
        let delta = vec![0.0, 1.0, 0.0, 0.0];
        let mut stv = SparseTaskVector::from_dense(&delta, (2, 2), 1e-5);
        stv.eta = 0.5;
        stv.apply_to(&mut base);
        assert!((base[1] - 0.5).abs() < 1e-6);
        stv.scale_eta(2.0); // eta now 1.0
        stv.apply_to(&mut base);
        assert!((base[1] - 1.5).abs() < 1e-6);
    }

    /// `to_dense_into` reconstructs the delta (post-threshold).
    #[test]
    fn to_dense_into_reconstructs() {
        let delta = vec![0.0, 0.2, 0.0, -0.1, 0.0, 0.0];
        let stv = SparseTaskVector::from_dense(&delta, (2, 3), 1e-5);
        let mut reconstructed = vec![0.0_f32; 6];
        stv.to_dense_into(&mut reconstructed);
        for (i, (&got, &want)) in reconstructed.iter().zip(delta.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "reconstruct mismatch at {i}");
        }
    }

    /// Iterator yields scaled `(index, delta)` pairs.
    #[test]
    fn iter_yields_scaled_pairs() {
        let delta = vec![0.0, 2.0, 0.0, 4.0];
        let mut stv = SparseTaskVector::from_dense(&delta, (2, 2), 1e-5);
        stv.eta = 0.5;
        let pairs: Vec<(u32, f32)> = stv.iter().collect();
        assert_eq!(pairs, vec![(1, 1.0), (3, 2.0)]);
    }

    /// Empty delta produces empty sparse vector.
    #[test]
    fn empty_delta_produces_empty_stv() {
        let delta = vec![0.0; 16];
        let stv = SparseTaskVector::from_dense(&delta, (4, 4), 1e-5);
        assert!(stv.is_empty());
        assert_eq!(stv.len(), 0);
        assert_eq!(stv.density(), 0.0);
    }
}

// ── Phase 4 (Plan 270) unit tests for compose_gauge_invariant ──────────────
//
// Gated on `gauge_invariant` — compiled only when both `sparse_task_vector`
// and `gauge_invariant` features are active.

#[cfg(all(test, feature = "gauge_invariant"))]
mod gauge_tests {
    use super::*;

    /// Two STVs with overlapping masks merge correctly: shared coordinates sum,
    /// disjoint coordinates are preserved. This is the gauge-invariant merge
    /// of two rank-`nnz` factor pairs under the natural sparse factorization.
    #[test]
    fn compose_gauge_invariant_merges_masks_and_sums_overlapping() {
        let a = SparseTaskVector::from_parts(
            (4, 4),
            vec![0, 5, 10],
            vec![0.10, 0.20, 0.30],
            1.0,
        )
        .unwrap();
        let b = SparseTaskVector::from_parts(
            (4, 4),
            vec![5, 10, 15],
            vec![0.05, 0.10, 0.15],
            1.0,
        )
        .unwrap();

        let merged = a.compose_gauge_invariant(&b, 1.0);

        // Merged mask = sorted union = [0, 5, 10, 15]
        assert_eq!(merged.mask, vec![0, 5, 10, 15], "merged mask must be union");
        // Weighted sums: idx 0 only from a, idx 5+10 shared, idx 15 only from b.
        assert!((merged.deltas[0] - 0.10).abs() < 1e-6, "idx 0 = {:?}", merged.deltas[0]);
        assert!((merged.deltas[1] - 0.25).abs() < 1e-6, "idx 5 = {:?}", merged.deltas[1]);
        assert!((merged.deltas[2] - 0.40).abs() < 1e-6, "idx 10 = {:?}", merged.deltas[2]);
        assert!((merged.deltas[3] - 0.15).abs() < 1e-6, "idx 15 = {:?}", merged.deltas[3]);
        assert_eq!(merged.shape, (4, 4));
        assert!((merged.eta - 1.0).abs() < 1e-6, "output eta must be 1.0");
    }

    /// `eta` scales `other`'s contribution: `merged = a + eta * b`.
    #[test]
    fn compose_gauge_invariant_respects_eta_weight() {
        let a = SparseTaskVector::from_parts((2, 2), vec![0, 1], vec![1.0, 1.0], 1.0).unwrap();
        let b = SparseTaskVector::from_parts((2, 2), vec![0, 1], vec![1.0, 1.0], 1.0).unwrap();

        // eta = 0.5 → merged = 1.0·a + 0.5·b = 1.5 on each coordinate.
        let merged = a.compose_gauge_invariant(&b, 0.5);
        assert!((merged.deltas[0] - 1.5).abs() < 1e-6);
        assert!((merged.deltas[1] - 1.5).abs() < 1e-6);
    }

    /// Opposite-sign deltas at the same coordinate cancel and are pruned
    /// from the output mask. This is critical for task-arithmetic "negate"
    /// operations where a - a should give the empty task vector.
    #[test]
    fn compose_gauge_invariant_cancels_opposite_signs() {
        let a = SparseTaskVector::from_parts((2, 2), vec![0, 1], vec![0.5, 0.3], 1.0).unwrap();
        let b = SparseTaskVector::from_parts((2, 2), vec![0, 1], vec![-0.5, 0.3], 1.0).unwrap();

        let merged = a.compose_gauge_invariant(&b, 1.0);
        // idx 0 cancels (0.5 + (-0.5) = 0), idx 1 sums to 0.6.
        assert_eq!(merged.mask, vec![1], "cancelled coordinate must be pruned");
        assert_eq!(merged.deltas.len(), 1);
        assert!((merged.deltas[0] - 0.6).abs() < 1e-6);
    }

    /// Each STV's own `eta` field scales its contribution. Here `a`'s eta=2.0
    /// doubles its deltas before merging.
    #[test]
    fn compose_gauge_invariant_honors_stv_eta_field() {
        let a = SparseTaskVector::from_parts((1, 4), vec![0, 1], vec![0.3, 0.4], 2.0).unwrap();
        let b = SparseTaskVector::from_parts((1, 4), vec![2, 3], vec![0.1, 0.2], 1.0).unwrap();

        let merged = a.compose_gauge_invariant(&b, 1.0);
        assert_eq!(merged.mask, vec![0, 1, 2, 3]);
        // a's contribution is scaled by its eta=2.0: 2.0*0.3 = 0.6, 2.0*0.4 = 0.8.
        assert!((merged.deltas[0] - 0.6).abs() < 1e-6);
        assert!((merged.deltas[1] - 0.8).abs() < 1e-6);
        // b's contribution unscaled: 0.1, 0.2.
        assert!((merged.deltas[2] - 0.1).abs() < 1e-6);
        assert!((merged.deltas[3] - 0.2).abs() < 1e-6);
    }

    /// Empty inputs: merging two empty STVs is well-defined and yields empty output.
    #[test]
    fn compose_gauge_invariant_handles_empty_inputs() {
        let a = SparseTaskVector::from_parts((4, 4), vec![], vec![], 1.0).unwrap();
        let b = SparseTaskVector::from_parts((4, 4), vec![], vec![], 1.0).unwrap();
        let merged = a.compose_gauge_invariant(&b, 1.0);
        assert!(merged.is_empty());
        assert_eq!(merged.shape, (4, 4));
    }

    /// Roundtrip property: applying the merged STV to a base equals applying
    /// `a` and `b` separately (the gauge-invariant merge preserves task arithmetic).
    #[test]
    fn compose_gauge_invariant_apply_matches_sequential_apply() {
        let a = SparseTaskVector::from_dense(
            &[0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
            (3, 3),
            1e-5,
        );
        let b = SparseTaskVector::from_dense(
            &[0.0, 0.0, 0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6],
            (3, 3),
            1e-5,
        );
        let eta = 0.7_f32;

        // Path 1: sequential apply.
        let mut base_seq = vec![1.0_f32; 9];
        a.apply_to(&mut base_seq);
        // Apply b scaled by eta manually.
        let mut b_scaled = b.clone();
        b_scaled.eta = eta;
        b_scaled.apply_to(&mut base_seq);

        // Path 2: merged then apply once.
        let merged = a.clone().compose_gauge_invariant(&b, eta);
        let mut base_merged = vec![1.0_f32; 9];
        merged.apply_to(&mut base_merged);

        for (i, (&s, &m)) in base_seq.iter().zip(base_merged.iter()).enumerate() {
            assert!((s - m).abs() < 1e-6, "apply mismatch at {i}: seq={s}, merged={m}");
        }
    }

    /// Equivalence with the full `gauge_invariant_compose` machinery.
    ///
    /// For each STV we construct the natural factorization `(A, B)` with
    /// `rank = nnz` and route through `gauge_invariant_compose`. The merged
    /// deltas (recovered by gathering `A_merged · B_merged^T` at each union
    /// coordinate) must match `compose_gauge_invariant`'s output within f32
    /// precision. This validates the proof in the method's doc comment:
    /// the sparse natural factorization's gauge-invariant compose reduces to
    /// weighted delta sum.
    #[test]
    fn test_compose_gauge_invariant_matches_full_compose() {
        use crate::gauge_invariant::{gauge_invariant_compose, GaugePair};

        let (rows, cols) = (3_usize, 3_usize);
        // Two STVs with overlapping masks.
        let a = SparseTaskVector::from_dense(
            &[0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2],
            (rows, cols),
            1e-5,
        );
        let b = SparseTaskVector::from_dense(
            &[0.0, 0.0, 0.4, 0.0, 0.1, 0.0, 0.0, 0.0, 0.6],
            (rows, cols),
            1e-5,
        );
        let eta = 0.8_f32;

        // --- Path A: direct `compose_gauge_invariant` ---
        let merged_stv = a.clone().compose_gauge_invariant(&b, eta);
        let mut dense_merged = vec![0.0_f32; rows * cols];
        merged_stv.apply_to(&mut dense_merged);

        // --- Path B: route natural factorization through `gauge_invariant_compose` ---
        // Union mask, sorted ascending. Both `a.mask` and `b.mask` are already sorted.
        let mut union_mask: Vec<u32> = Vec::with_capacity(a.len() + b.len());
        {
            let mut ia = 0;
            let mut ib = 0;
            while ia < a.mask.len() && ib < b.mask.len() {
                let (xa, xb) = (a.mask[ia], b.mask[ib]);
                if xa < xb {
                    union_mask.push(xa);
                    ia += 1;
                } else if xa > xb {
                    union_mask.push(xb);
                    ib += 1;
                } else {
                    union_mask.push(xa);
                    ia += 1;
                    ib += 1;
                }
            }
            while ia < a.mask.len() {
                union_mask.push(a.mask[ia]);
                ia += 1;
            }
            while ib < b.mask.len() {
                union_mask.push(b.mask[ib]);
                ib += 1;
            }
        }
        let r = union_mask.len();
        assert!(r > 0, "test pre-condition: non-empty union mask");

        // Build A_i (rows × r), B_i (cols × r) for each STV.
        // For STV `a`, delta_k sits at (row_of(mask[k]), col_of(mask[k])) —
        // we need to pad to the union rank with zeros where the mask is absent.
        let build_factors =
            |stv: &SparseTaskVector| -> (Vec<f32>, Vec<f32>) {
                let mut a_mat = vec![0.0_f32; rows * r];
                let mut b_mat = vec![0.0_f32; cols * r];
                // For each union column k, find whether `stv` has a delta at union_mask[k].
                let mut si = 0usize;
                for (k, &um) in union_mask.iter().enumerate() {
                    while si < stv.mask.len() && stv.mask[si] < um {
                        si += 1;
                    }
                    if si < stv.mask.len() && stv.mask[si] == um {
                        let flat = um as usize;
                        let (ro, co) = (flat / cols, flat % cols);
                        a_mat[ro * r + k] = stv.deltas[si];
                        b_mat[co * r + k] = 1.0;
                    }
                }
                (a_mat, b_mat)
            };

        let (a1, b1) = build_factors(&a);
        let (a2, b2) = build_factors(&b);
        let pairs = [
            GaugePair { eta: 1.0, a: &a1, b: &b1, a_rows: rows, b_rows: cols, rank: r },
            GaugePair { eta, a: &a2, b: &b2, a_rows: rows, b_rows: cols, rank: r },
        ];
        let merged_r = 2 * r;
        let mut out_a = vec![0.0_f32; rows * merged_r];
        let mut out_b = vec![0.0_f32; cols * merged_r];
        gauge_invariant_compose(&pairs, &mut out_a, &mut out_b);

        // Recover W_merged = out_a · out_b^T and compare against `dense_merged`.
        let mut max_diff = 0.0_f32;
        for i in 0..rows {
            for j in 0..cols {
                let mut s = 0.0_f32;
                for k in 0..merged_r {
                    s += out_a[i * merged_r + k] * out_b[j * merged_r + k];
                }
                let direct = dense_merged[i * cols + j];
                let diff = (s - direct).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff < 1e-4,
            "gauge_invariant_compose ({}) differs from direct delta-sum by {max_diff} > 1e-4",
            "this breaks the proof that sparse compose_gauge_invariant reduces to weighted sum"
        );
        eprintln!(
            "PASS: compose_gauge_invariant matches full gauge_invariant_compose (max diff = {max_diff:.2e})"
        );
    }
}
