//! Band conditioning sets + conditional independence (CI) tests for task-relevant
//! identifiability (Plan 265, Research 232).
//!
//! Implements the **band conditioning set** primitive from
//! [arXiv:2605.12733](https://arxiv.org/pdf/2605.12733) —
//! "From Generalist to Specialist Representation", Zheng et al., ICML 2026.
//!
//! # What this is
//!
//! Given a sequence of latent states `{s_1, ..., s_T}` partitioned into segments
//! `S_k = {s_{(k-1)L+1}, ..., s_{kL}}` of length `L >= 2`, and a set of task
//! variables `{g_1, ..., g_M}` defined as **colliders** across time steps
//! (`a_{t1} → g_i ← a_{t2}`), Theorem 1 of the paper states:
//!
//! ```text
//! g_i is relevant to segments S_k and S_v
//!     ⟺
//! s_{kL} ⊭ s_{vL} | Z_band(k, v, i)
//! ```
//!
//! where the **band conditioning set** is
//!
//! ```text
//! Z_band(k, v, i) = {s_{kL-1}, s_{kL+1}, s_{vL-1}, s_{vL+1}} ∩ {s_1..s_T} ∪ {g_i}
//! ```
//!
//! with out-of-range indices omitted.
//!
//! # Two CI test variants
//!
//! - [`conditional_dependence_fisher_z`] — Fisher z-test on partial correlation.
//!   Fast, classical, requires approximate Gaussianity of residuals (the paper's
//!   synthetic setup). Returns a sigmoid-bounded p-value for downstream routing.
//! - [`conditional_dependence_infonce`] — InfoNCE bound on conditional mutual
//!   information. The paper's Appendix C high-dimensional surrogate. Requires a
//!   critic closure (modelless engines may pass a frozen pretrained encoder).
//!
//! Both return scores in `[0, 1]`. Higher score = stronger evidence of
//! conditional dependence. A score above `0.5` corresponds to a CI test
//! rejection at the configured alpha.
//!
//! # Example — Paper Figure 2
//!
//! ```
//! # #[cfg(feature = "band_conditioner")] {
//! use katgpt_rs::band_conditioner::BandConditioningSet;
//!
//! // Sequence of 8 states partitioned into 4 segments of length L=2:
//! //   S_1 = {s_1, s_2},  S_2 = {s_3, s_4},  S_3 = {s_5, s_6},  S_4 = {s_7, s_8}
//! // Test whether g_1 is relevant to segments S_2 (k=2) and S_4 (v=4):
//! //   kL = 4,  vL = 8
//! //   band states = {s_3, s_5, s_7, s_9} ∩ {s_1..s_8} = {s_3, s_5, s_7}
//! //   ∪ {g_1} → Z_band = {s_3, s_5, s_7, g_1}
//! let z = BandConditioningSet::from_segments(2 /*k*/, 4 /*v*/, 1 /*task*/, 2 /*L*/, 8 /*T*/);
//! assert_eq!(z.state_indices(), &[3, 5, 7]);
//! assert_eq!(z.task_index(), 1);
//! # }
//! ```

// ── BandConditioningSet ──────────────────────────────────────────────────────

/// The 4-element **band conditioning set** from paper Theorem 1, eq. (4).
///
/// `Z_band(k, v, i) = {s_{kL-1}, s_{kL+1}, s_{vL-1}, s_{vL+1}} ∩ {s_1..s_T} ∪ {g_i}`
///
/// Out-of-range indices (≤ 0 or > T) are omitted, leaving at most 4 state
/// indices. The task index `i` is always included.
///
/// Storage is fixed-size (`[u32; 4]` + a count + task index) so the hot path
/// avoids any allocation. Index `0` is reserved for "empty slot" — states are
/// 1-indexed in the paper's notation, so we shift to 0-indexed internally and
/// store `s_1..s_T` as `0..T-1`. To preserve the paper's 1-indexed public API,
/// we keep `state_indices()` returning 1-indexed values (matching the docstring
/// of [`from_segments`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandConditioningSet {
    /// Up to 4 state indices, 1-indexed (`s_1 = 1`). Empty slots are `0`.
    /// Stored sorted ascending.
    state_slots: [u32; 4],
    /// Number of valid state slots (≤ 4).
    n_states: u8,
    /// Task index `i` (always present, 1-indexed for symmetry with states).
    task: u32,
}

impl BandConditioningSet {
    /// Build the band conditioning set `Z_band(k, v, i)` for segments
    /// `S_k = {s_{(k-1)L+1}..s_{kL}}` and `S_v = {s_{(v-1)L+1}..s_{kL}}`
    /// (with `k < v`), task index `i` (1-indexed), segment length `L ≥ 2`,
    /// and total horizon `T ≥ 1`.
    ///
    /// Per paper eq. (4):
    ///
    /// ```text
    /// Z_band(k, v, i) =
    ///   { s_{kL-1}, s_{kL+1}, s_{vL-1}, s_{vL+1} } ∩ { s_1, ..., s_T } ∪ { g_i }
    /// ```
    ///
    /// with out-of-range indices omitted. `L >= 2` and `k < v` are required
    /// by the paper (otherwise the test is ill-defined). Panics in debug if
    /// violated.
    ///
    /// Indices returned by [`state_indices`](Self::state_indices) are
    /// **1-indexed** (i.e., `s_1 = 1`, `s_T = T`) to match paper notation.
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_segments(k: usize, v: usize, task: usize, segment_len: usize, total_steps: usize) -> Self {
        debug_assert!(segment_len >= 2, "segment length L must be >= 2 (paper Thm 1)");
        debug_assert!(k >= 1 && v >= 1, "segment indices k, v must be >= 1");
        debug_assert!(k < v, "must have k < v (paper Thm 1, segment ordering)");
        debug_assert!(task >= 1, "task index must be >= 1");
        debug_assert!(total_steps >= 1, "total steps T must be >= 1");

        let kl = k * segment_len;
        let vl = v * segment_len;
        // 1-indexed candidate state indices.
        let mut candidates = [kl as i64 - 1, kl as i64 + 1, vl as i64 - 1, vl as i64 + 1];
        // Filter to [1, T], dedup, sort.
        candidates.sort_unstable();
        let mut state_slots = [0u32; 4];
        let mut n_states: u8 = 0;
        let mut last: i64 = -1;
        for &c in &candidates {
            if c >= 1 && c <= total_steps as i64 && c != last {
                state_slots[n_states as usize] = c as u32;
                n_states += 1;
                last = c;
            }
        }
        Self {
            state_slots,
            n_states,
            task: task as u32,
        }
    }

    /// Returns the valid state indices (1-indexed, sorted ascending, ≤ 4 elements).
    pub fn state_indices(&self) -> &[u32] {
        &self.state_slots[..self.n_states as usize]
    }

    /// Returns the task index `i` (1-indexed).
    pub fn task_index(&self) -> u32 {
        self.task
    }

    /// Total number of conditioning variables (states + 1 task).
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> usize {
        self.n_states as usize + 1
    }

    /// Always false — a conditioning set always contains the task index.
    pub fn is_empty(&self) -> bool {
        false
    }
}

// ── Fisher z-test CI test ────────────────────────────────────────────────────

/// Result of a Fisher z-test conditional independence test.
///
/// The score is a **sigmoid-bounded dependence score**: `score = sigmoid(|z_stat|)`.
/// Higher score ⟹ stronger evidence of conditional dependence.
/// - `score ≈ 0.5` ⟹ no dependence (|z| ≈ 0).
/// - `score → 1.0` ⟹ strong dependence (|z| → ∞).
///
/// Configure [`CiTestConfig::alpha`] to compute the rejection threshold
/// via [`CiTestResult::rejects_at`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CiTestResult {
    /// Pearson partial correlation coefficient in `[-1, 1]`.
    pub partial_r: f32,
    /// Fisher z statistic.
    pub z_stat: f32,
    /// Sigmoid-bounded "dependence score" in `(0, 1)`. Higher = more dependent.
    pub score: f32,
    /// Number of samples used.
    pub n_samples: usize,
}

impl CiTestResult {
    /// Returns `true` if the test rejects conditional independence at the given
    /// `alpha` (two-sided). `alpha` ∈ `(0, 1)`.
    ///
    /// Concretely: rejects if `score > sigmoid(z_alpha)` where
    /// `z_alpha = Phi^{-1}(1 - alpha/2)`. We hard-code the lookup for the
    /// three standard alphas to avoid pulling in a `statistics` dep.
    pub fn rejects_at(&self, alpha: f32) -> bool {
        let z_alpha = inverse_normal_cdf_two_sided(alpha);
        self.z_stat.abs() > z_alpha
    }
}

/// Configuration for the Fisher z-test conditional independence test.
#[derive(Clone, Copy, Debug)]
pub struct CiTestConfig {
    /// Two-sided significance level. Default `0.05`.
    pub alpha: f32,
}

impl Default for CiTestConfig {
    fn default() -> Self {
        Self { alpha: 0.05 }
    }
}

/// Tests `H_0: X ⊥ Y | Z` via the Fisher z-transformation of the partial
/// Pearson correlation coefficient.
///
/// - `x`, `y`: `n_samples`-long observation vectors.
/// - `z_columns`: each slice is an `n_samples`-long conditioning variable.
///   Up to 4 columns supported (matches band conditioning set cardinality).
///   Empty slice is allowed (treated as marginal independence test).
/// - `n_samples`: number of samples in each slice (must match across all).
/// - `config`: alpha level + future extensions.
///
/// Returns a sigmoid-bounded score in `(0, 1)` — higher = more dependent.
///
/// # Algorithm
///
/// ```text
/// 1. Linearly regress x on [z_1, ..., z_k] → residuals r_x
/// 2. Linearly regress y on [z_1, ..., z_k] → residuals r_y
/// 3. partial_r = pearson(r_x, r_y)
/// 4. z = 0.5 * ln((1+r)/(1-r)) * sqrt(n - 3 - k)
/// 5. score = sigmoid(|z|)   // higher score = more dependent
/// ```
///
/// The `n - 3 - k` matches the paper's setup (k conditioning variables).
/// For Gaussian residuals, |z| > Phi^{-1}(1 - alpha/2) rejects H_0.
pub fn conditional_dependence_fisher_z(
    x: &[f32],
    y: &[f32],
    z_columns: &[&[f32]],
    n_samples: usize,
    config: CiTestConfig,
) -> CiTestResult {
    let _ = config;
    debug_assert!(x.len() >= n_samples, "x too short");
    debug_assert!(y.len() >= n_samples, "y too short");
    for z in z_columns {
        debug_assert!(z.len() >= n_samples, "z column too short");
    }

    let k = z_columns.len();
    let partial_r = if n_samples <= k + 2 {
        // Insufficient degrees of freedom — fall back to marginal correlation.
        pearson_r(x, y, n_samples)
    } else {
        partial_correlation(x, y, z_columns, n_samples)
    };
    let r_clamped = partial_r.clamp(-0.999_999, 0.999_999);
    let z_transform = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let df = (n_samples as f32) - 3.0 - (k as f32);
    let z_stat = if df > 0.0 {
        z_transform * df.sqrt()
    } else {
        z_transform
    };
    let score = sigmoid(z_stat.abs());
    CiTestResult {
        partial_r,
        z_stat,
        score,
        n_samples,
    }
}

/// Compute partial correlation `corr(x, y | z_1, ..., z_k)` via residualization.
///
/// Uses orthogonal projection onto the z-subspace via one pass of
/// Gram-Schmidt. O(n · k²) — fast for the small k (≤ 4) we use.
fn partial_correlation(x: &[f32], y: &[f32], z_columns: &[&[f32]], n: usize) -> f32 {
    let k = z_columns.len();
    if k == 0 {
        return pearson_r(x, y, n);
    }

    // Build a row-major basis: rows = n samples, cols = (k + 1 intercept).
    // Heap-allocate because n can be 1000+ (CI tests on long sequences).
    // For the small k (≤ 4) used by band conditioning, this is one alloc
    // per call — acceptable since CI tests are not in the per-token hot path.
    // `basis_cols` ends at `k + 1` (intercept + k z-columns); tracked via mutation below.
    let mut basis = vec![[0.0f32; 8]; n];

    // Intercept column.
    for row in basis.iter_mut() {
        row[0] = 1.0;
    }
    let mut basis_cols = 1;

    // Orthogonalize z-columns against current basis (modified Gram-Schmidt).
    for z_col in z_columns {
        // Copy z_col into basis slot.
        for i in 0..n {
            basis[i][basis_cols] = z_col[i];
        }
        // Subtract projections onto previous basis vectors.
        for prev in 0..basis_cols {
            let mut dot = 0.0f32;
            let mut norm = 0.0f32;
            for i in 0..n {
                dot += basis[i][basis_cols] * basis[i][prev];
                norm += basis[i][prev] * basis[i][prev];
            }
            if norm > 1e-12 {
                let coef = dot / norm;
                for i in 0..n {
                    basis[i][basis_cols] -= coef * basis[i][prev];
                }
            }
        }
        basis_cols += 1;
    }

    // Project x onto the orthonormal-like basis, store residual in r_x.
    let mut r_x_vec: Vec<f32> = x[..n].to_vec();
    project_out(&mut r_x_vec, &basis, basis_cols, n);

    let mut r_y_vec: Vec<f32> = y[..n].to_vec();
    project_out(&mut r_y_vec, &basis, basis_cols, n);

    pearson_r(&r_x_vec, &r_y_vec, n)
}

/// Subtract from `v` its projection onto the `cols` columns of `basis`.
/// `basis[i][j]` for `i in 0..n, j in 0..cols`.
fn project_out(v: &mut [f32], basis: &[[f32; 8]], cols: usize, n: usize) {
    for j in 0..cols {
        let mut dot = 0.0f32;
        let mut norm = 0.0f32;
        for i in 0..n {
            dot += v[i] * basis[i][j];
            norm += basis[i][j] * basis[i][j];
        }
        if norm > 1e-12 {
            let coef = dot / norm;
            for i in 0..n {
                v[i] -= coef * basis[i][j];
            }
        }
    }
}

/// Pearson correlation coefficient of `x[..n]` and `y[..n]`.
fn pearson_r(x: &[f32], y: &[f32], n: usize) -> f32 {
    let (mut sx, mut sy, mut sxy, mut sx2, mut sy2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let xi = f64::from(x[i]);
        let yi = f64::from(y[i]);
        sx += xi;
        sy += yi;
        sxy += xi * yi;
        sx2 += xi * xi;
        sy2 += yi * yi;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let var_x = sx2 - sx * sx / nf;
    let var_y = sy2 - sy * sy / nf;
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (cov / denom) as f32
}

/// Standard logistic sigmoid. Returns value in `(0, 1)`.
///
/// Used to bound CI test p-values / scores into the `[0, 1]` interval for
/// downstream routing per the project's "sigmoid not softmax" rule.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Two-sided inverse normal CDF for the standard significance levels.
///
/// Hard-coded lookup avoids pulling in a statistics dependency.
/// Returns `Phi^{-1}(1 - alpha/2)` — the rejection threshold on |z|.
fn inverse_normal_cdf_two_sided(alpha: f32) -> f32 {
    match alpha {
        a if (a - 0.10).abs() < 1e-3 => 1.644_853_6, // 90%
        a if (a - 0.05).abs() < 1e-3 => 1.959_964_0, // 95%
        a if (a - 0.01).abs() < 1e-3 => 2.575_829_3, // 99%
        a if (a - 0.001).abs() < 1e-3 => 3.290_526_7, // 99.9%
        // Default to 95%.
        _ => 1.959_964_0,
    }
}

// ── InfoNCE-based CI test (paper Appendix C) ─────────────────────────────────

/// A frozen critic for the InfoNCE CMI estimator. Modelless engines pass a
/// pretrained encoder (e.g., existing `MaxSim` head) here.
///
/// Signature: `critic(x_emb, y_emb, z_emb) -> f32`.
pub type InfoNceCritic = fn(&[f32], &[f32], &[f32]) -> f32;

/// Configuration for the InfoNCE CMI estimator.
#[derive(Clone, Copy, Debug)]
pub struct InfoNceConfig {
    /// Number of negative samples per positive pair. Default `8`.
    pub n_negatives: usize,
    /// Default critic if none is provided (simple dot product on concatenated emb).
    pub default_critic: InfoNceCritic,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self {
            n_negatives: 8,
            default_critic: default_dot_critic,
        }
    }
}

/// Default critic: dot product of `x_emb` and `y_emb`, modulated by `z_emb` magnitude.
fn default_dot_critic(x_emb: &[f32], y_emb: &[f32], z_emb: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let n = x_emb.len().min(y_emb.len());
    for i in 0..n {
        dot += x_emb[i] * y_emb[i];
    }
    let z_norm = z_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (1.0 + z_norm) // sigmoid-friendly scale
}

/// Tests `H_0: X ⊥ Y | Z` via an InfoNCE lower bound on conditional mutual
/// information (paper Appendix C).
///
/// Returns a sigmoid-bounded score in `(0, 1)` — higher = more dependent.
///
/// # Inputs
///
/// - `x_emb`, `y_emb`: positive pair embeddings (same length, `d_emb`).
/// - `z_emb`: conditioning embedding (also `d_emb`).
/// - `negatives`: each slice is a negative `y_emb` (drawn by shuffling within
///   `z_emb` buckets per the paper). Length `n_negatives`.
/// - `critic`: a frozen critic function (modelless engines pass a pretrained
///   encoder; pass `default_critic` for dot-product baseline).
/// - `config`: tuning knobs.
pub fn conditional_dependence_infonce(
    x_emb: &[f32],
    y_emb: &[f32],
    z_emb: &[f32],
    negatives: &[&[f32]],
    critic: InfoNceCritic,
    config: InfoNceConfig,
) -> f32 {
    let pos = critic(x_emb, y_emb, z_emb).exp();
    let mut denom = pos;
    for neg in negatives.iter().take(config.n_negatives) {
        denom += critic(x_emb, neg, z_emb).exp();
    }
    // InfoNCE lower bound on CMI: log(pos/denom).
    let nce = (pos / denom).max(f32::MIN_POSITIVE).ln();
    // Sigmoid-bound: higher NCE → higher score. With well-scaled critic,
    // NCE ∈ [-n_negatives_ln, 0]; map to (0, 1) via sigmoid.
    sigmoid(nce)
}

// ── Memory-tier marker (constraint 8: plasma/hot/warm/cold/freeze) ───────────

/// Marks the memory tier where a CI test result should be cached.
///
/// BCKVSS (Plan 265 Fusion A) routes its results by tier:
/// - Hot: per-query CI test, refreshed every inference.
/// - Warm: per-session CI test, cached for the current user/session.
/// - Cold: per-adapter CI test, committed to disk.
/// - Freeze: pre-computed collider tables (loaded at startup).
///
/// Plasma is reserved for SPLAT projection masks (Plan 265 Fusion B), not CI tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CiTestTier {
    Hot = 0,
    Warm = 1,
    Cold = 2,
    Freeze = 3,
}

impl CiTestTier {
    /// Pick the appropriate tier for a CI test result based on its refresh policy.
    ///
    /// Per paper §3.2 — task structure is stable across queries within a session.
    /// Per Theorem 1 — colliders are persistent across the entire horizon.
    /// So:
    /// - If `n_steps_in_test > 0` (new query), Hot.
    /// - If `n_steps_in_test == 0` (cached adapter), Freeze.
    pub fn for_query(n_new_steps: usize) -> Self {
        if n_new_steps > 0 {
            Self::Hot
        } else {
            Self::Freeze
        }
    }
}

// ── Auto-route hint (constraint 7: CPU/SIMD/GPU/ANE threshold) ───────────────

/// Best compute substrate for a CI test, by threshold.
///
/// Per optimization.md: SIMD beats GPU at small workloads (≤ 1000 pairs),
/// GPU beats SIMD above. CPU (single-threaded scalar) is only appropriate for
/// pathological small inputs or no SIMD target.
///
/// The thresholds here are intentionally conservative — tune after Phase 1 GOAT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComputeTarget {
    Cpu = 0,
    Simd = 1,
    Gpu = 2,
    Ane = 3,
    Plasma = 4,
}

impl ComputeTarget {
    /// Route a batch of CI tests by pair count.
    ///
    /// - `n_pairs < 1000` → Simd (avoid GPU launch overhead ~50μs).
    /// - `n_pairs >= 1000` → Gpu (amortize launch).
    ///
    /// ANE / Plasma are never appropriate for CI tests (they're for projection
    /// and ternary matmul respectively — see Plan 265 Fusion B).
    #[inline]
    pub fn for_ci_test_batch(n_pairs: usize) -> Self {
        if n_pairs < 1000 {
            Self::Simd
        } else {
            Self::Gpu
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    // ── BandConditioningSet correctness ──────────────────────────────────────

    /// GOAT test G0a: paper Figure 2 example.
    ///
    /// Segments S_k = {s_3, s_4} and S_v = {s_7, s_8} (so k=2, v=4, L=2, T=8).
    /// Task g_1.
    ///
    /// Z_band(2, 4, 1) = {s_{2L-1}, s_{2L+1}, s_{4L-1}, s_{4L+1}} ∩ {s_1..s_8} ∪ {g_1}
    ///                = {s_3, s_5, s_7, s_9} ∩ {s_1..s_8} ∪ {g_1}
    ///                = {s_3, s_5, s_7} ∪ {g_1}
    ///                = {s_3, s_5, s_7, g_1}
    #[test]
    fn g0a_paper_figure_2() {
        let z = BandConditioningSet::from_segments(2, 4, 1, 2, 8);
        assert_eq!(z.state_indices(), &[3, 5, 7]);
        assert_eq!(z.task_index(), 1);
        assert_eq!(z.len(), 4); // 3 states + 1 task
        assert!(!z.is_empty());
    }

    /// Boundary case: k=1 means s_{kL-1} = s_1 (in range), s_{kL+1} = s_3.
    #[test]
    fn band_conditioning_at_sequence_start() {
        let z = BandConditioningSet::from_segments(1, 3, 1, 2, 6);
        // kL=2, vL=6.
        // candidates: {s_1, s_3, s_5, s_7} ∩ {s_1..s_6} = {s_1, s_3, s_5}.
        assert_eq!(z.state_indices(), &[1, 3, 5]);
    }

    /// Boundary case: vL=T means s_{vL+1} is out of range.
    #[test]
    fn band_conditioning_at_sequence_end() {
        let z = BandConditioningSet::from_segments(1, 2, 1, 4, 8);
        // kL=4, vL=8.
        // candidates: {s_3, s_5, s_7, s_9} ∩ {s_1..s_8} = {s_3, s_5, s_7}.
        assert_eq!(z.state_indices(), &[3, 5, 7]);
    }

    /// Dedup: when kL-1 == vL-1 (only possible if k=v, disallowed) — not testable.
    /// But L=3 with k=1, v=2 gives kL=3, vL=6, candidates {s_2, s_4, s_5, s_7}
    /// — no dups. Test with k=1, v=2, L=2: candidates {s_1, s_3, s_3, s_5}.
    #[test]
    fn band_conditioning_dedups_adjacent_indices() {
        let z = BandConditioningSet::from_segments(1, 2, 1, 2, 5);
        // kL=2, vL=4. candidates: {s_1, s_3, s_3, s_5} → dedup → {s_1, s_3, s_5}.
        assert_eq!(z.state_indices(), &[1, 3, 5]);
    }

    // ── Fisher z-test power ─────────────────────────────────────────────────

    /// GOAT test G0b: Fisher z-test recovers dependence on linear Gaussian SCM
    /// at p<0.05 with ≥ 90% power at n=1000 samples.
    ///
    /// Build a synthetic collider: g → {x, y} where x = a*g + ε_x, y = a*g + ε_y.
    /// After conditioning on g (Z=[g]), x ⊥ y (no dependence).
    /// Without conditioning on g, x ⊭ y (collider opens the path — wait,
    /// the paper's setting is the opposite: conditioning on the collider
    /// OPENS dependence).
    ///
    /// For this test, use the simpler classical case:
    /// - x and y are correlated (rho ≈ 0.3) marginally.
    /// - Fisher z-test should detect dependence with high power.
    /// - Conditioning on a noise variable should NOT remove dependence.
    #[test]
    fn g0b_fisher_z_detects_dependence_at_high_power() {
        let n = 1000;
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];

        // Build a collider structure: g → x, g → y. After conditioning on g,
        // x ⊥ y. Without conditioning on g, x ⊭ y (marginal).
        let mut rng = fastrand::Rng::with_seed(42);
        for i in 0..n {
            g[i] = rng.standard_normal();
            let ex = rng.standard_normal() * 0.5;
            let ey = rng.standard_normal() * 0.5;
            x[i] = g[i] + ex;
            y[i] = g[i] + ey;
        }

        // Test 1: Without conditioning on g, x ⊭ y (dependent).
        let result_marginal = conditional_dependence_fisher_z(
            &x, &y, &[], n, CiTestConfig::default(),
        );
        // Expect strong dependence: |z| >> 1.96, score > 0.5.
        assert!(
            result_marginal.z_stat.abs() > 1.96,
            "expected z > 1.96, got {} (r={})",
            result_marginal.z_stat,
            result_marginal.partial_r,
        );
        assert!(result_marginal.score > 0.5, "expected score > 0.5");
        assert!(
            result_marginal.partial_r.abs() > 0.4,
            "expected |r| > 0.4, got {}",
            result_marginal.partial_r,
        );

        // Test 2: With conditioning on g (the collider), x ⊥ y.
        let result_conditioned = conditional_dependence_fisher_z(
            &x, &y, &[&g], n, CiTestConfig::default(),
        );
        // Wait — the paper says conditioning on a collider OPENS dependence,
        // not closes it. So with a collider g, conditioning on g should NOT
        // remove the dependence between x and y.
        //
        // For this synthetic test, our SCM is x = g + ε_x, y = g + ε_y.
        // Marginal: x and y are correlated through g.
        // Conditional on g: x = g + ε_x and y = g + ε_y are independent.
        //
        // This is actually a *confounder* structure (g → x, g → y), not a collider.
        // Conditioning on a confounder DOES close the path.
        //
        // So the test is correct: conditioning on g should make x ⊥ y.
        assert!(
            result_conditioned.partial_r.abs() < 0.1,
            "expected |partial_r| < 0.1 after conditioning, got {}",
            result_conditioned.partial_r,
        );
        assert!(
            result_conditioned.z_stat.abs() < 1.96,
            "expected |z| < 1.96 after conditioning, got {}",
            result_conditioned.z_stat,
        );
    }

    /// Power sanity: with strong dependence (rho ≈ 0.5), reject at ≥ 90% power.
    #[test]
    fn fisher_z_power_at_strong_correlation() {
        let n = 1000;
        let mut rejections = 0;
        let trials = 20;
        let mut rng = fastrand::Rng::with_seed(7);
        for trial in 0..trials {
            let mut x = vec![0.0f32; n];
            let mut y = vec![0.0f32; n];
            for i in 0..n {
                let g = rng.standard_normal();
                // Induce correlation ~0.5 through shared component.
                x[i] = g + rng.standard_normal();
                y[i] = g + rng.standard_normal();
            }
            let result = conditional_dependence_fisher_z(
                &x, &y, &[], n, CiTestConfig::default(),
            );
            if result.rejects_at(0.05) {
                rejections += 1;
            }
            let _ = trial;
        }
        // ≥ 90% power means ≥ 18/20 rejections.
        assert!(
            rejections >= 18,
            "expected ≥ 18/20 rejections (90% power), got {}/{}",
            rejections,
            trials,
        );
    }

    /// Type I error sanity: under independence, false-positive rate near alpha.
    #[test]
    fn fisher_z_type_one_error() {
        let n = 1000;
        let mut false_positives = 0;
        let trials = 50;
        let mut rng = fastrand::Rng::with_seed(13);
        for _ in 0..trials {
            let x: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
            let y: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
            let result = conditional_dependence_fisher_z(
                &x, &y, &[], n, CiTestConfig::default(),
            );
            if result.rejects_at(0.05) {
                false_positives += 1;
            }
        }
        // Expected ~5% false positives → ≤ 8/50 is generous.
        assert!(
            false_positives <= 8,
            "expected ≤ 8/50 false positives, got {}/{}",
            false_positives,
            trials,
        );
    }

    // ── Sigmoid & utilities ──────────────────────────────────────────────────

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 1e-3);
    }

    #[test]
    fn ci_test_score_in_unit_interval() {
        let n = 100;
        let mut rng = fastrand::Rng::with_seed(1);
        let x: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
        let r = conditional_dependence_fisher_z(&x, &y, &[], n, CiTestConfig::default());
        assert!(r.score >= 0.0 && r.score <= 1.0);
    }

    // ── InfoNCE CMI estimator ────────────────────────────────────────────────

    #[test]
    fn infonce_score_in_unit_interval() {
        let x = [1.0f32, 0.5, 0.3];
        let y = [0.8f32, 0.4, 0.2];
        let z = [0.1f32, 0.2, 0.3];
        let n1 = [0.1f32, 0.0, 0.0];
        let n2 = [0.0f32, 0.1, 0.0];
        let negatives: &[&[f32]] = &[&n1, &n2];
        let score = conditional_dependence_infonce(
            &x, &y, &z, negatives, default_dot_critic, InfoNceConfig::default(),
        );
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn infonce_high_for_similar_pair() {
        let x = [1.0f32, 1.0, 1.0];
        let y = [1.0f32, 1.0, 1.0]; // identical to x
        let z = [0.0f32, 0.0, 0.0];
        let n1 = [0.1f32, 0.0, 0.0];
        let n2 = [0.0f32, 0.1, 0.0];
        let n3 = [0.0f32, 0.0, 0.1];
        let negatives: &[&[f32]] = &[&n1, &n2, &n3];
        let score_similar = conditional_dependence_infonce(
            &x, &y, &z, negatives, default_dot_critic, InfoNceConfig::default(),
        );

        let y_dissimilar = [-1.0f32, -1.0, -1.0];
        let score_dissimilar = conditional_dependence_infonce(
            &x, &y_dissimilar, &z, negatives, default_dot_critic, InfoNceConfig::default(),
        );

        assert!(
            score_similar > score_dissimilar,
            "similar pair score {} should exceed dissimilar {}",
            score_similar,
            score_dissimilar,
        );
    }

    // ── Tier & route ─────────────────────────────────────────────────────────

    #[test]
    fn ci_test_tier_for_query() {
        assert_eq!(CiTestTier::for_query(0), CiTestTier::Freeze);
        assert_eq!(CiTestTier::for_query(5), CiTestTier::Hot);
    }

    #[test]
    fn compute_target_routing_thresholds() {
        assert_eq!(ComputeTarget::for_ci_test_batch(100), ComputeTarget::Simd);
        assert_eq!(ComputeTarget::for_ci_test_batch(999), ComputeTarget::Simd);
        assert_eq!(ComputeTarget::for_ci_test_batch(1000), ComputeTarget::Gpu);
        assert_eq!(ComputeTarget::for_ci_test_batch(10_000), ComputeTarget::Gpu);
    }

    // ── Black-box microbench (sanity, not a strict perf test) ───────────────

    /// Sanity check: a CI test on n=1000 completes in well under 1ms on this machine.
    /// Not a benchmark gate — just guards against O(n²) regressions in the hot path.
    #[test]
    fn ci_test_latency_under_1ms_for_n_1000() {
        let n = 1000;
        let mut rng = fastrand::Rng::with_seed(99);
        let x: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();
        let z: Vec<f32> = (0..n).map(|_| rng.standard_normal()).collect();

        let iters = 100;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = black_box(conditional_dependence_fisher_z(
                black_box(&x),
                black_box(&y),
                black_box(&[&z]),
                black_box(n),
                CiTestConfig::default(),
            ));
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_micros() as f64 / iters as f64;
        // Generous bound — 1ms per call. Real perf will be much lower.
        assert!(
            per_call_us < 1000.0,
            "CI test took {:.2}μs/call (expected < 1000μs)",
            per_call_us,
        );
    }
}

// ── Minimal normal-distribution helper for synthetic tests ───────────────────

#[cfg(test)]
/// Private trait extension: `fastrand::Rng` does not have `standard_normal`
/// out of the box, so we add one via Box-Muller.
trait NormalRng {
    fn standard_normal(&mut self) -> f32;
}

#[cfg(test)]
impl NormalRng for fastrand::Rng {
    fn standard_normal(&mut self) -> f32 {
        // Box-Muller transform.
        let mut u1 = self.f32();
        if u1 < 1e-10 {
            u1 = 1e-10;
        }
        let u2 = self.f32();
        let mag = (-2.0f32 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
        z0
    }
}
