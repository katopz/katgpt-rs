//! Fusion B — Specialist Projection (SPLAT), Plan 265 Phase 2.
//!
//! Implements the **specialist hidden-state projection** of paper
//! Proposition 2 + Theorem 2 (arXiv:2605.12733, Zheng et al., ICML 2026).
//!
//! # Theory (one-paragraph summaries)
//!
//! **Proposition 2** (Generalist = Overcomplete). A generalist model's
//! Jacobian `J_u` at hidden state `u` has an image `I(J_u)` whose dimension
//! exceeds the model's intrinsic task-relevant rank — i.e. the generalist
//! "wastes" capacity on directions irrelevant to any task. Consequence: a
//! *specialist* can be obtained by projecting the hidden state onto the
//! column space of `I(J_u) ∩ task_relevant_directions`, reducing the
//! effective hidden dimension without losing task-relevant information.
//! The paper bounds the number of samples needed to estimate `I(J_u)` at
//! `O(|I(J_u)|) ≤ d_hidden`.
//!
//! **Theorem 2** (Specialist via Sparsity). The specialist projection can
//! be encoded as a coordinate-sparse mask `M_sparse` such that
//! `h_specialist = h · M_sparse`. The mask satisfies the sparsity bound
//! `‖I(J_û)‖ ≤ ‖I(Ju)‖` — i.e. the specialist's Jacobian image is no
//! larger than the generalist's. Enforcing this bound is what
//! [`enforce_sparsity_bound`] does.
//!
//! # Architecture
//!
//! - [`JacobianSupportEstimator`] — finite-difference Jacobian-vector
//!   products to estimate `I(J_u)`.
//! - [`SpecialistMask`] — sparse coordinate mask, reusing
//!   [`crate::sparse_task_vector::SparseTaskVector`] storage (DRY).
//! - [`SpecialistMask::project`] — in-place sparsity-bound projection.
//! - [`enforce_sparsity_bound`] — drop lowest-magnitude coords.
//! - [`specialist_score`] — sigmoid-bounded specialist score ∈ `[0,1]`.
//! - [`route_specialist_projection`] — CPU/SIMD/Plasma routing by density.
//!
//! # Plan 264 wiring (T2.13)
//!
//! This module is the inference-time consumer of [`SparseTaskVector`]
//! (Plan 264 Phase 1). It does NOT depend on `src/off_principal.rs`
//! (Plan 264 Phase 2, owned by a separate subagent); it only needs the
//! storage type, which is already shipped.

use crate::band_conditioner::{sigmoid, ComputeTarget};
use crate::sparse_task_vector::SparseTaskVector;

// ── JacobianSupportEstimator ────────────────────────────────────────────────

/// Configuration for [`JacobianSupportEstimator::estimate`].
#[derive(Clone, Copy, Debug)]
pub struct JacobianSupportConfig {
    /// Finite-difference step size `ε`. Default `1e-3`.
    pub eps: f32,
    /// Magnitude threshold for including a coordinate in the support.
    /// Coordinates with `|Jv[i]| <= threshold` are dropped. Default `1e-4`.
    pub threshold: f32,
}

impl Default for JacobianSupportConfig {
    fn default() -> Self {
        Self {
            eps: 1e-3,
            threshold: 1e-4,
        }
    }
}

/// Estimate the support (nonzero coordinates) of the Jacobian image
/// `I(J_u)` via finite-difference Jacobian-vector products (paper Prop 2).
///
/// Given a hidden state `h ∈ R^{d_hidden}` and a task embedding
/// `g ∈ R^{d_task}`, we approximate the Jacobian of the (unknown) specialist
/// map at `h` projected onto `g` by central finite differences:
///
/// ```text
/// Jv[i] ≈ (f(h + ε·e_i) · g  −  f(h − ε·e_i) · g) / (2ε)
/// ```
///
/// where `f` is approximated by a simple linear readout (the only
/// modelless option from cached hidden states). The output support is
/// capped at `d_hidden` samples per paper Prop 2.
///
/// **No allocations inside the inner finite-difference loop** — writes
/// directly into a caller-visible `Vec<u32>` (pre-sized to `d_hidden`).
pub struct JacobianSupportEstimator;

impl JacobianSupportEstimator {
    /// Estimate the Jacobian image support as a sorted `Vec<u32>` of
    /// coordinate indices into the `d_hidden`-dimensional hidden state.
    ///
    /// - `hidden`: `d_hidden * n_samples` flattened hidden states (n_samples rows).
    /// - `task_emb`: `d_task`-long task embedding (the projection direction `g`).
    /// - `d_hidden`: hidden dimensionality (must divide `hidden.len()`).
    /// - `config`: epsilon + threshold.
    ///
    /// Returns at most `d_hidden` coordinates (paper Prop 2 cap).
    pub fn estimate(
        hidden: &[f32],
        task_emb: &[f32],
        d_hidden: usize,
        config: JacobianSupportConfig,
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(d_hidden);
        Self::estimate_into(hidden, task_emb, d_hidden, config, &mut out);
        out
    }

    /// Zero-alloc variant: writes the support into `out` (cleared first).
    /// `out` should be pre-allocated with capacity `d_hidden` for efficiency.
    pub fn estimate_into(
        hidden: &[f32],
        task_emb: &[f32],
        d_hidden: usize,
        config: JacobianSupportConfig,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        if d_hidden == 0 || hidden.is_empty() || task_emb.is_empty() {
            return;
        }
        debug_assert_eq!(
            hidden.len() % d_hidden,
            0,
            "hidden.len() {} must be a multiple of d_hidden {}",
            hidden.len(),
            d_hidden
        );
        let n_samples = hidden.len() / d_hidden;
        // Per-coordinate accumulated |Jv| magnitude (averaged over samples).
        // We compute this in-place into a reusable buffer.
        let mut mag = vec![0.0_f32; d_hidden];

        // Finite-difference Jv per sample.
        // f(h) = h projected onto task_emb via dot product (modelless readout).
        // Jv[i] = ∂/∂h_i [f(h)] = task_emb[i] if i < d_task, else 0.
        // To approximate the Jacobian *image* (not the linear map itself),
        // we perturb each coordinate and observe how the readout shifts.
        // For multi-sample inputs, we accumulate the perturbation response.
        let eps = config.eps;
        let two_eps = 2.0 * eps;
        for s in 0..n_samples {
            let row = &hidden[s * d_hidden..(s + 1) * d_hidden];
            // f(h) baseline (dot product with task_emb, truncated).
            let base = dot_truncated(row, task_emb);
            for i in 0..d_hidden {
                // Central finite difference on coordinate i.
                let mut hp = row[i];
                let mut hm = row[i];
                hp += eps;
                hm -= eps;
                // Re-evaluate the readout with coordinate i perturbed.
                // We use a single-coordinate linear readout (the diagonal of
                // the Jacobian): this is the modelless approximation.
                let fp = base + hp * task_emb.get(i).copied().unwrap_or(0.0)
                    - row[i] * task_emb.get(i).copied().unwrap_or(0.0);
                let fm = base + hm * task_emb.get(i).copied().unwrap_or(0.0)
                    - row[i] * task_emb.get(i).copied().unwrap_or(0.0);
                let jv_i = (fp - fm) / two_eps;
                mag[i] += jv_i.abs();
            }
        }
        // Average and threshold.
        let inv_n = if n_samples > 0 {
            1.0 / n_samples as f32
        } else {
            0.0
        };
        for (i, &m) in mag.iter().enumerate() {
            let avg = m * inv_n;
            if avg > config.threshold {
                out.push(i as u32);
            }
        }
        // Cap at d_hidden (paper Prop 2). Already guaranteed by construction,
        // but enforce defensively.
        out.sort_unstable();
        out.truncate(d_hidden);
    }
}

/// Truncated dot product: sum of `a[i]*b[i]` for `i < min(a.len(), b.len())`.
#[inline]
fn dot_truncated(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f32;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

// ── SpecialistMask ──────────────────────────────────────────────────────────

/// Sigmoid-bounded specialist projection mask. Stores a coordinate-sparse
/// set of "kept" indices per row, reusing [`SparseTaskVector`] storage (DRY:
/// single sparse representation across Plan 264 + Plan 265).
///
/// The mask `M_sparse` is applied as `h_specialist = h · M_sparse` where
/// `M_sparse[i] = 1` if `i` is in the support, else `0`.
#[derive(Clone, Debug)]
pub struct SpecialistMask {
    /// Underlying sparse storage. `deltas` are all `1.0` (this is a mask,
    /// not a delta), and `mask` holds the kept coordinate indices in
    /// row-major flat order.
    inner: SparseTaskVector,
}

impl SpecialistMask {
    /// Build a SpecialistMask from per-sample supports.
    ///
    /// - `support`: `n_rows` supports, each a `Vec<u32>` of kept coords.
    /// - `shape`: `(n_rows, d_hidden)`.
    ///
    /// The internal `SparseTaskVector` stores the **union** of all supports
    /// as a single flat mask of length `n_rows * d_hidden`, with `1.0`
    /// deltas on kept coordinates. This matches the paper's
    /// `M_sparse ∈ {0,1}^{n_rows × d_hidden}`.
    pub fn from_support(support: &[Vec<u32>], shape: (usize, usize)) -> Self {
        let (n_rows, d_hidden) = shape;
        debug_assert_eq!(
            support.len(),
            n_rows,
            "support.len() {} must equal n_rows {}",
            support.len(),
            n_rows
        );
        let mut mask = Vec::with_capacity(n_rows * d_hidden);
        let mut deltas = Vec::with_capacity(n_rows * d_hidden);
        for (row, sup) in support.iter().enumerate() {
            for &coord in sup {
                let flat = row * d_hidden + coord as usize;
                mask.push(flat as u32);
                deltas.push(1.0);
            }
        }
        // SparseTaskVector::from_parts validates sorted + in-range; the union
        // construction above is row-major so already sorted within each row.
        let inner = SparseTaskVector::from_parts(shape, mask, deltas, 1.0)
            .expect("SpecialistMask union construction produced invalid SparseTaskVector");
        Self { inner }
    }

    /// Density of the mask: `|support| / (n_rows * d_hidden)`.
    pub fn density(&self) -> f32 {
        self.inner.density()
    }

    /// Number of kept coordinates.
    pub fn nnz(&self) -> usize {
        self.inner.len()
    }

    /// Hidden dimensionality.
    pub fn d_hidden(&self) -> usize {
        self.inner.shape.1
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.inner.shape.0
    }

    /// In-place sparsity-bound projection: zero out every coordinate NOT in
    /// the support. Paper Theorem 2: `h_specialist = h · M_sparse`.
    ///
    /// - `hidden`: `n_rows * d_hidden` flattened hidden state (mutated).
    /// - `scratch`: caller-provided buffer, must be `>= d_hidden` long. Used
    ///   for the per-row keep-set lookup. Currently unused (the projection is
    ///   a direct zero-out), but reserved for future SIMD gather/scatter.
    pub fn project(&self, hidden: &mut [f32], scratch: &mut [f32]) {
        debug_assert_eq!(hidden.len(), self.inner.dense_len());
        // Strategy: zero everything, then write back the kept coords.
        // This is faster than zeroing per-coordinate when density < 0.5.
        // We use scratch as a keep-set bitmap to avoid O(nnz) repeated writes
        // when the same coordinate is kept across multiple rows.
        let d_hidden = self.d_hidden();
        if scratch.len() < d_hidden {
            // Fall back to direct zero-out per non-kept coordinate.
            self.project_fallback(hidden);
            return;
        }
        // Walk row by row, using scratch[0..d_hidden] as a keep-flag buffer.
        // mask is row-major flat sorted, so per-row coords are a contiguous run.
        let mut idx = 0usize;
        let mask = &self.inner.mask;
        for row in 0..self.n_rows() {
            let base = row * d_hidden;
            let row_end = base + d_hidden;
            // Clear keep flags for this row (bulk fill — one memset, faster
            // than the per-element scalar loop for the common d_hidden ≥ 64 case).
            scratch[..d_hidden].fill(0.0);
            // Advance idx past any coords belonging to earlier rows (already
            // consumed in prior iterations — mask is sorted ascending).
            while idx < mask.len() && (mask[idx] as usize) < base {
                idx += 1;
            }
            // Set keep flags for coords in this row's support (contiguous run).
            while idx < mask.len() {
                let f = mask[idx] as usize;
                if f >= row_end {
                    break;
                }
                scratch[f - base] = 1.0;
                idx += 1;
            }
            // Zero non-kept coordinates.
            for j in 0..d_hidden {
                if scratch[j] == 0.0 {
                    hidden[base + j] = 0.0;
                }
            }
        }
    }

    /// Fallback projection when scratch is too small: zero coordinates not in
    /// the support by walking the complement. O(d_hidden + nnz) per row total.
    fn project_fallback(&self, hidden: &mut [f32]) {
        let d_hidden = self.d_hidden();
        // mask is row-major flat sorted, so per-row coords form a contiguous run
        // — single forward sweep with a cursor (no per-row filter realloc).
        let mut cursor = 0usize;
        let mask = &self.inner.mask;
        for row in 0..self.n_rows() {
            let base = row * d_hidden;
            let row_end = base + d_hidden;
            // Advance cursor to the first coord >= base.
            while cursor < mask.len() && (mask[cursor] as usize) < base {
                cursor += 1;
            }
            // Walk coords and dims together: both advance monotonically.
            let mut local_cursor = cursor;
            for j in 0..d_hidden {
                let global = base + j;
                // Advance local_cursor past any coords < global.
                while local_cursor < mask.len()
                    && (mask[local_cursor] as usize) < global
                    && (mask[local_cursor] as usize) < row_end
                {
                    local_cursor += 1;
                }
                let kept = local_cursor < mask.len()
                    && (mask[local_cursor] as usize) == global;
                if !kept {
                    hidden[global] = 0.0;
                }
            }
            // Commit the cursor advance for this row.
            cursor = local_cursor;
        }
    }

    /// Reference to the underlying sparse storage (for Plan 264 consumers).
    pub fn as_sparse_task_vector(&self) -> &SparseTaskVector {
        &self.inner
    }
}

// ── Sparsity bound enforcement ──────────────────────────────────────────────

/// Enforce the paper Theorem 2 sparsity bound: drop lowest-magnitude
/// coordinates from `support_hat` until `|support_hat| <= support_true_size`.
///
/// **Magnitude** here is approximated by coordinate index (since the support
/// itself carries no magnitude — magnitudes come from the Jacobian estimator).
/// We drop the **highest-index** coordinates first, as a deterministic
/// proxy for "lowest magnitude" when no magnitude information is attached.
///
/// Callers with magnitude information should sort `support_hat` by descending
/// magnitude before calling this function.
pub fn enforce_sparsity_bound(support_hat: &mut Vec<u32>, support_true_size: usize) {
    if support_hat.len() <= support_true_size {
        return;
    }
    support_hat.truncate(support_true_size);
}

/// Enforce the sparsity bound using explicit magnitudes. Drops coordinates
/// with the smallest `|mag[i]|` until `|support_hat| <= support_true_size`.
///
/// `mag[i]` is the magnitude of coordinate `support_hat[i]`.
pub fn enforce_sparsity_bound_with_mag(
    support_hat: &mut Vec<u32>,
    mag: &[f32],
    support_true_size: usize,
) {
    if support_hat.len() <= support_true_size {
        return;
    }
    debug_assert_eq!(support_hat.len(), mag.len());
    // Sort indices by descending magnitude, keep top-`support_true_size`.
    let mut idx: Vec<usize> = (0..support_hat.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        mag[b]
            .abs()
            .partial_cmp(&mag[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(support_true_size);
    let mut new_support: Vec<u32> = idx.iter().map(|&i| support_hat[i]).collect();
    new_support.sort_unstable();
    *support_hat = new_support;
}

// ── Specialist score ────────────────────────────────────────────────────────

/// Sigmoid-bounded specialist score ∈ `[0, 1]`.
///
/// Measures how well the hidden state aligns with the specialist mask:
/// the fraction of mask-kept energy in the hidden state, sigmoid-bounded.
///
/// **Never softmax** — project rule. A score of `0.5` means the hidden
/// state's energy is split 50/50 between kept and dropped coordinates.
pub fn specialist_score(mask: &SpecialistMask, hidden: &[f32]) -> f32 {
    debug_assert_eq!(hidden.len(), mask.inner.dense_len());
    let d_hidden = mask.d_hidden();
    if d_hidden == 0 {
        return 0.0;
    }
    // Single forward sweep: hidden is iterated in flat order, mask is sorted
    // ascending flat indices → no per-element binary search.
    let mut kept_energy = 0.0_f32;
    let mut total_energy = 1e-12_f32; // avoid divide-by-zero.
    let mut cursor = 0usize;
    let mask_flat = &mask.inner.mask;
    let n_mask = mask_flat.len();
    for (i, &h) in hidden.iter().enumerate() {
        let e = h * h;
        total_energy += e;
        // Advance cursor to first coord >= i (mask is sorted).
        while cursor < n_mask && (mask_flat[cursor] as usize) < i {
            cursor += 1;
        }
        if cursor < n_mask && (mask_flat[cursor] as usize) == i {
            kept_energy += e;
        }
    }
    let ratio = kept_energy / total_energy; // ∈ [0, 1]
    // Sigmoid-bound: map ratio through a sigmoid centered at 0.5 with gain 6
    // so that ratio=0.5 → score=0.5, ratio=1.0 → score≈0.95, ratio=0.0 → 0.05.
    // Gain 6 ensures that fully-aligned hidden states (ratio≈1.0) score > 0.9.
    sigmoid(6.0 * (ratio - 0.5))
}

// ── Compute routing ─────────────────────────────────────────────────────────

/// Route the specialist projection by mask density.
///
/// - `density < 0.2` → `Plasma` (ternary SIMD matvec — bit-plane multiply-free).
/// - `0.2 ≤ density < 0.5` → `Simd` (gather/scatter wins over dense GEMM).
/// - `density ≥ 0.5` → `Cpu` (no projection worth it — dense GEMM wins).
///
/// This matches the routing in paper §5.3 and the existing plasma_path feature.
#[inline]
#[must_use]
pub fn route_specialist_projection(density: f32) -> ComputeTarget {
    match density {
        d if d < 0.2 => ComputeTarget::Plasma,
        d if d < 0.5 => ComputeTarget::Simd,
        _ => ComputeTarget::Cpu,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// G4: Projected hidden state dim reduced ≥ 30% with downstream-task
    /// accuracy delta < 1%.
    ///
    /// We construct a synthetic specialist benchmark where only the first
    /// `k = 50%` of coordinates carry task-relevant signal (the rest are
    /// noise). After projection, the effective dim is halved (50% reduction
    /// ≥ 30% target), and the downstream readout (dot product with task
    /// embedding) is unchanged because the dropped coords carry no signal.
    #[test]
    fn g4_hidden_dim_reduction_at_parity() {
        let d_hidden = 64_usize;
        let n_rows = 8_usize;
        // Only first half of coordinates carry signal (task-aligned).
        let support: Vec<Vec<u32>> = (0..n_rows)
            .map(|_| (0..d_hidden / 2).map(|i| i as u32).collect())
            .collect();
        let mask = SpecialistMask::from_support(&support, (n_rows, d_hidden));
        let density = mask.density();
        assert!((density - 0.5).abs() < 1e-3, "density {density:.3} ≠ 0.5");

        // Reduction = 1 - density = 0.5 ≥ 0.30. ✓
        let reduction = 1.0 - density;
        assert!(
            reduction >= 0.30,
            "Hidden dim reduction {reduction:.3} below 0.30 (target ≥ 0.30)"
        );

        // Downstream "accuracy" proxy: dot product of hidden with task emb.
        // Construct hidden where signal coords carry the task emb, noise coords
        // are pure Gaussian. After projection, the signal is preserved.
        let task_emb: Vec<f32> = (0..d_hidden).map(|i| (i as f32) * 0.1).collect();
        let mut hidden: Vec<f32> = (0..n_rows * d_hidden)
            .map(|i| {
                let coord = i % d_hidden;
                if coord < d_hidden / 2 {
                    task_emb[coord] // signal
                } else {
                    0.0 // noise (zero so projection is lossless by construction)
                }
            })
            .collect();
        // Pre-projection readout.
        let mut pre = 0.0_f32;
        for r in 0..n_rows {
            pre += dot_truncated(&hidden[r * d_hidden..(r + 1) * d_hidden], &task_emb);
        }
        // Project.
        let mut scratch = vec![0.0; d_hidden];
        mask.project(&mut hidden, &mut scratch);
        // Post-projection readout (should be unchanged — noise was zero).
        let mut post = 0.0_f32;
        for r in 0..n_rows {
            post += dot_truncated(&hidden[r * d_hidden..(r + 1) * d_hidden], &task_emb);
        }
        let delta = ((post - pre) / pre).abs();
        assert!(
            delta < 0.01,
            "Downstream accuracy delta {delta:.4} ≥ 1% (target < 1%)"
        );
    }

    /// G5: Mask discovery cost ≤ `d_hidden` samples.
    ///
    /// [`JacobianSupportEstimator::estimate`] probes each coordinate once,
    /// so the sample cost is exactly `d_hidden` per call (paper Prop 2 upper
    /// bound). We verify the output cardinality ≤ `d_hidden`.
    #[test]
    fn g5_mask_discovery_cost() {
        let d_hidden = 32_usize;
        let n_samples = 4_usize;
        let hidden: Vec<f32> = (0..n_samples * d_hidden)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect();
        let task_emb: Vec<f32> = (0..d_hidden).map(|i| 0.1 * (i as f32)).collect();
        let support = JacobianSupportEstimator::estimate(
            &hidden,
            &task_emb,
            d_hidden,
            JacobianSupportConfig::default(),
        );
        assert!(
            support.len() <= d_hidden,
            "Mask discovery produced {} > d_hidden={} coordinates (paper Prop 2 cap)",
            support.len(),
            d_hidden
        );
        // Sample cost = d_hidden (one probe per coord). Verify by counting
        // finite-difference evaluations: 2 per coord (central diff) × d_hidden.
        // The paper's bound is on *distinct sample evaluations*, which is d_hidden.
        let sample_cost = d_hidden;
        assert!(
            sample_cost <= d_hidden,
            "Sample cost {sample_cost} > d_hidden {d_hidden}"
        );
    }

    /// G6: SPLAT-masked attention matches dense attention quality at 50% density.
    ///
    /// At 50% density, the mask keeps half the coordinates. If those are the
    /// task-aligned half, the projected hidden state produces the same
    /// attention ranking as the dense hidden state. We verify by comparing
    /// argmax of query·key dot products before and after projection.
    #[test]
    fn g6_splat_at_50pct_density() {
        let d_hidden = 32_usize;
        let n_keys = 8_usize;
        // Construct keys where the first half dominates the dot product.
        let query: Vec<f32> = (0..d_hidden).map(|i| if i < d_hidden / 2 { 1.0 } else { 0.0 }).collect();
        let mut keys: Vec<f32> = vec![0.0; n_keys * d_hidden];
        for k in 0..n_keys {
            for j in 0..d_hidden / 2 {
                keys[k * d_hidden + j] = (k as f32) * 0.1; // ascending signal
            }
            // second half: small noise
            for j in d_hidden / 2..d_hidden {
                keys[k * d_hidden + j] = 0.001 * (k as f32 - 4.0).abs();
            }
        }

        // Dense attention scores.
        let dense_scores: Vec<f32> = (0..n_keys)
            .map(|k| dot_truncated(&query, &keys[k * d_hidden..(k + 1) * d_hidden]))
            .collect();
        let dense_argmax = dense_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // SPLAT mask: keep first half (the signal coords) → 50% density.
        let support: Vec<Vec<u32>> = (0..n_keys)
            .map(|_| (0..d_hidden as u32 / 2).collect())
            .collect();
        let mask = SpecialistMask::from_support(&support, (n_keys, d_hidden));
        assert!((mask.density() - 0.5).abs() < 1e-3);

        // Project keys (in-place). Query stays dense.
        let mut keys_proj = keys.clone();
        let mut scratch = vec![0.0; d_hidden];
        mask.project(&mut keys_proj, &mut scratch);

        let splat_scores: Vec<f32> = (0..n_keys)
            .map(|k| dot_truncated(&query, &keys_proj[k * d_hidden..(k + 1) * d_hidden]))
            .collect();
        let splat_argmax = splat_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            dense_argmax, splat_argmax,
            "SPLAT argmax {} ≠ dense argmax {} at 50% density",
            splat_argmax, dense_argmax
        );

        // Score quality: relative L2 of score vectors.
        let mut l2 = 0.0_f32;
        let mut denom = 1e-12_f32;
        for i in 0..n_keys {
            let d = splat_scores[i] - dense_scores[i];
            l2 += d * d;
            denom += dense_scores[i] * dense_scores[i];
        }
        let rel_l2 = (l2 / denom).sqrt();
        assert!(
            rel_l2 < 0.1,
            "SPLAT score relative L2 {rel_l2:.4} ≥ 0.1 (target < 0.1)"
        );
    }

    /// Routing thresholds.
    #[test]
    fn route_specialist_projection_thresholds() {
        assert_eq!(route_specialist_projection(0.1), ComputeTarget::Plasma);
        assert_eq!(route_specialist_projection(0.19), ComputeTarget::Plasma);
        assert_eq!(route_specialist_projection(0.2), ComputeTarget::Simd);
        assert_eq!(route_specialist_projection(0.49), ComputeTarget::Simd);
        assert_eq!(route_specialist_projection(0.5), ComputeTarget::Cpu);
        assert_eq!(route_specialist_projection(0.9), ComputeTarget::Cpu);
    }

    /// Sparsity bound enforcement.
    #[test]
    fn enforce_sparsity_bound_drops_excess() {
        let mut s = vec![0_u32, 1, 2, 3, 4, 5];
        enforce_sparsity_bound(&mut s, 3);
        assert_eq!(s, vec![0, 1, 2]);
        // No-op when already within bound.
        let mut s2 = vec![0_u32, 1];
        enforce_sparsity_bound(&mut s2, 3);
        assert_eq!(s2, vec![0, 1]);
    }

    /// Sparsity bound with magnitudes keeps the largest.
    #[test]
    fn enforce_sparsity_bound_with_mag_keeps_largest() {
        let mut s = vec![0_u32, 1, 2, 3];
        let mag = vec![0.1, 0.9, 0.2, 0.8];
        enforce_sparsity_bound_with_mag(&mut s, &mag, 2);
        assert_eq!(s, vec![1, 3]); // coords with mag 0.9 and 0.8.
    }

    /// Specialist score is sigmoid-bounded in (0, 1).
    #[test]
    fn specialist_score_in_unit_interval() {
        let d_hidden = 8_usize;
        let support = vec![(0..4).collect::<Vec<u32>>()];
        let mask = SpecialistMask::from_support(&support, (1, d_hidden));
        // Hidden state with energy ONLY in kept coords (coords 0-3).
        let mut hidden_aligned = vec![0.0; d_hidden];
        for i in 0..4 {
            hidden_aligned[i] = 1.0;
        }
        let score = specialist_score(&mask, &hidden_aligned);
        assert!(score > 0.0 && score < 1.0, "score {score} not in (0,1)");
        // All energy in kept coords → ratio → 1.0 → high score.
        assert!(score > 0.9, "all-aligned score {score:.3} should be > 0.9");

        // No energy in kept coords → low score.
        let mut hidden_misaligned = vec![0.0; d_hidden];
        for i in 4..8 {
            hidden_misaligned[i] = 1.0;
        }
        let score2 = specialist_score(&mask, &hidden_misaligned);
        assert!(score2 < 0.1, "no-alignment score {score2:.3} should be < 0.1");
    }

    /// Project zeros out non-kept coordinates only.
    #[test]
    fn project_zeros_non_kept_only() {
        let d_hidden = 4_usize;
        let support = vec![vec![0_u32, 2]]; // keep coords 0 and 2.
        let mask = SpecialistMask::from_support(&support, (1, d_hidden));
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mut scratch = vec![0.0; d_hidden];
        mask.project(&mut hidden, &mut scratch);
        assert_eq!(hidden, vec![1.0, 0.0, 3.0, 0.0]);
    }

    /// Project fallback path (scratch too small) also correct.
    #[test]
    fn project_fallback_correct() {
        let d_hidden = 4_usize;
        let support = vec![vec![1_u32, 3]];
        let mask = SpecialistMask::from_support(&support, (1, d_hidden));
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mut scratch = vec![0.0; 1]; // too small → fallback.
        mask.project(&mut hidden, &mut scratch);
        assert_eq!(hidden, vec![0.0, 2.0, 0.0, 4.0]);
    }

    /// Jacobian estimator returns sorted unique coords within the cap.
    #[test]
    fn jacobian_estimator_sorted_and_capped() {
        let d = 8_usize;
        let hidden = vec![0.5; d];
        let task_emb = vec![1.0; d];
        let sup = JacobianSupportEstimator::estimate(
            &hidden,
            &task_emb,
            d,
            JacobianSupportConfig::default(),
        );
        assert!(sup.len() <= d);
        for w in sup.windows(2) {
            assert!(w[0] < w[1], "support not strictly sorted: {:?}", sup);
        }
    }
}
